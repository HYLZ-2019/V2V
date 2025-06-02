# Note: This is deprecated code. I stopped using it because the simple ESIM noise model resulted in better results than the complex V2E noise model -- maybe because models would overfit to the information-providing noise, maybe due to non-optimal parameter settings, or maybe I did something else wrong. It was never tested with the WebVid dataset (I used other worse-quality video datasets in early research) and it doesn't fit with the current code. Anyway, I'll keep it here in case it can be useful in the future.

"""
DVS simulator.
Compute event voxels from input frames.
Most of the code is adapted from https://github.com/SensorsINI/v2e/blob/master/v2ecore/emulator.py.
Modified by HYLZ-2019:
I removed code for voxel-to-event.
I removed code for output, logging, visualization and the csdvs & scidvs models. 
I removed code for photoreceptor noise, and improved the shot noise generation.
I changed sampling model of threshold.
This version runs on CPU in numpy. (v2v_core_torch.py runs on GPU in Torch.)
"""
import math
import os
import pickle
import random
import cv2
import h5py
import numpy as np
from typing import Optional, Any
import numpy as np

def viz_voxel(voxel):
	'''
	voxel: np.array, [num_frames, height, width]
	'''
	maxval = np.percentile(voxel, 99)
	minval = np.percentile(voxel, 1)
	maxval = max(maxval, 1)
	minval = min(minval, -1)
	voxel = np.clip(voxel, minval, maxval)
	voxel = np.where(voxel > 0, voxel/maxval, -voxel/minval)
	voxel = (voxel + 1) / 2 * 255
	voxel = voxel.astype(np.uint8)
	return voxel

def rotate_cov(theta, cov):
	rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
	return np.dot(rot, np.dot(cov, rot.T))

def compute_event_map(diff_frame, pos_thres, neg_thres):
	"""
		Compute event maps, i.e. 2d arrays of [width,height] containing quantized number of ON and OFF events.

	Args:
		diff_frame:  the input difference frame between stored log intensity and current frame log intensity [width, height]
		pos_thres:  ON threshold values [width, height]
		neg_thres:  OFF threshold values [width, height]

	Returns:
		pos_evts_frame, neg_evts_frame;  2d arrays of integer ON and OFF event counts
	"""
	# extract positive and negative differences
	pos_frame = np.clip(diff_frame, a_min=0, a_max=None)
	neg_frame = np.clip(-diff_frame, a_min=0, a_max=None)

	# compute quantized number of ON and OFF events for each pixel
	pos_evts_frame = np.floor_divide(pos_frame, pos_thres)
	neg_evts_frame = np.floor_divide(neg_frame, neg_thres)

	return pos_evts_frame, neg_evts_frame


def generate_shot_noise(
		shot_noise_rate_hz,
		delta_time,
		shot_noise_inten_factor,
		inten01,
		pos_thres_pre_prob,
		neg_thres_pre_prob):
	"""Generate shot noise.
	:param shot_noise_rate_hz: the rate per pixel in hz
	:param delta_time: the delta time for this frame in seconds
	:param shot_noise_inten_factor: factor to model the slight increase
		of shot noise with intensity when shot noise dominates at low intensity
	:param inten01: the pixel light intensities in this frame; shape is used to generate output
	:param pos_thres_pre_prob: per pixel factor to generate more
		noise from pixels with lower ON threshold: self.pos_thres_nominal/self.pos_thres
	:param neg_thres_pre_prob: same for OFF

	:returns: [H, W] map of ON and OFF events
	"""

	# shot noise factor is the expectation of the number of OFF events in this frame (which is tiny typically)
	# we compute it by taking half the total shot noise rate (OFF only),
	# multiplying by the delta time of this frame,
	# and multiplying by the intensity factor

	inten_factor = 1 - (1 - shot_noise_inten_factor) * inten01 # According to inten01, it decreases from 1 to shot_noise_inten_factor=0.25
	pos_factor = inten_factor * pos_thres_pre_prob # Those with smaller thresholds get more shot noise expectation
	pos_pix_factor = pos_factor / np.mean(pos_factor)  # Normalize to make the total number of noise events correct. No worry for division by zero, because all pos_factor elements are >0.
	neg_factor = inten_factor * neg_thres_pre_prob
	neg_pix_factor = neg_factor / np.mean(neg_factor)  

	shot_noise_factor = (shot_noise_rate_hz/2)*delta_time

	pos_exp = pos_pix_factor * shot_noise_factor
	neg_exp = neg_pix_factor * shot_noise_factor

	# Let's assume the number of shot noise events on each pixel follows a poisson distribution. To make a pixel to expect exp events, the poissoin distribution parameter is set to exp.
	pos_shot_noise = np.random.poisson(pos_exp)
	neg_shot_noise = np.random.poisson(neg_exp)
	
	return pos_shot_noise, neg_shot_noise


def lin_log(x, threshold=20):
	"""
	linear mapping + logarithmic mapping.

	:param x: float or ndarray
		the input linear value in range 0-255 TODO assumes 8 bit
	:param threshold: float threshold 0-255
		the threshold for transition from linear to log mapping

	Returns: the log value
	"""
	# converting x into np.float64.
	if x.dtype is not np.float64:  # note float64 to get rounding to work
		x = x.astype(np.float64)

	f = (1./threshold) * math.log(threshold)

	y = np.where(x <= threshold, x*f, np.log(x+1e-9))

	# important, we do a floating point round to some digits of precision
	# to avoid that adding threshold and subtracting it again results
	# in different number because first addition shoots some bits off
	# to never-never land, thus preventing the OFF events
	# that ideally follow ON events when object moves by
	rounding = 1e8
	y = np.round(y*rounding)/rounding
 
	y = np.log(x/255 + 0.01)

	return y.astype(np.float32)

def low_pass_filter(
		log_new_frame,
		lp_log_frame,
		inten01,
		delta_time,
		cutoff_hz=0):
	"""Compute intensity-dependent low-pass filter.

	# Arguments
		log_new_frame: new frame in lin-log representation.
		lp_log_frame:
		inten01: the scaling of filter time constant array, or None to not scale
		delta_time:
		cutoff_hz:

	# Returns
		new_lp_log_frame
	"""
	if cutoff_hz <= 0:
		# unchanged
		return log_new_frame

	# else low pass
	tau = 1/(math.pi*2*cutoff_hz)

	# make the update proportional to the local intensity
	# the more intensity, the shorter the time constant
	if inten01 is not None:
		eps = inten01*(delta_time/tau)
	else:
		eps=delta_time/tau
	eps = np.clip(eps, a_min=None, a_max=1)  # keep filter stable

	# first internal state is updated
	new_lp_log_frame = (1-eps)*lp_log_frame+eps*log_new_frame

	# then 2nd internal state (output) is updated from first
	# Note that observations show that one pole is nearly always dominant,
	# so the 2nd stage is just copy of first stage

	# (1-eps)*self.lpLogFrame1+eps*self.lpLogFrame0 # was 2nd-order,
	# now 1st order.

	return new_lp_log_frame

def rescale_intensity_frame(new_frame):
	"""Rescale intensity frames.

	make sure we get no zero time constants
	limit max time constant to ~1/10 of white intensity level
	"""
	return (new_frame+20)/275.

def subtract_leak_current(base_log_frame,
						  leak_rate_hz,
						  delta_time,
						  pos_thres,
						  leak_jitter_fraction,
						  noise_rate_array):
	"""Subtract leak current from base log frame."""

	# np.random.randn is a normal distribution with mean 0 and variance 1
	rand = np.random.randn(noise_rate_array.shape[0], noise_rate_array.shape[1])

	curr_leak_rate = \
		leak_rate_hz*noise_rate_array*(1-leak_jitter_fraction*rand)

	delta_leak = delta_time*curr_leak_rate*pos_thres  # this is a matrix

	# ideal model
	#  delta_leak = delta_time*leak_rate_hz*pos_thres  # this is a matrix

	return base_log_frame-delta_leak



class EventEmulator(object):
	# frames that can be displayed and saved to video with their plotting/display settings
	l255 = np.log(255)
	gr = (0, 255)  # display as 8 bit int gray image
	lg = (0, l255)  # display as log image with max ln(255)
	slg = (
		-l255 / 8,
		l255 / 8)  # display as signed log image with 1/8 of full scale for better visibility of faint contrast
	MODEL_STATES = {'new_frame': gr, 'log_new_frame': lg,
					'lp_log_frame': lg, 'scidvs_highpass': slg, 'photoreceptor_noise_arr': slg, 'cs_surround_frame': lg,
					'c_minus_s_frame': slg, 'base_log_frame': slg, 'diff_frame': slg}

	MAX_CHANGE_TO_TERMINATE_EULER_SURROUND_STEPPING = 1e-5

	SINGLE_PIXEL_STATES_FILENAME='pixel-states.dat'
	SINGLE_PIXEL_MAX_SAMPLES=10000
 
	def log_thres_to_thres(self, thres):
		return np.exp(thres) - 1e-3

	def thres_to_log_thres(self, thres):
		return np.log(thres + 1e-3)

	def __init__(
			self,
			threshold_model: str = "pn_related",
			thres_mean_mean: float = 0.5,
			thres_mean_std: float = 0.1,
			thres_diff_mean: float = 0,
			thres_diff_std: float = 0.1,
			cutoff_hz: float = 0.0,
			leak_rate_hz: float = 0.1,
			refractory_period_s: float = 0.0,
			shot_noise_rate_hz: float = 0.0,  # rate in hz of temporal noise events
			leak_jitter_fraction: float = 0.1,
			noise_rate_cov_decades: float = 0.1,
			seed: int = None,
	):
		"""
		Parameters
		----------
		pos_thres: float, default 0.21
			nominal threshold of triggering positive event in log intensity.
		neg_thres: float, default 0.17
			nominal threshold of triggering negative event in log intensity.
		sigma_thres: float, default 0.03
			std deviation of threshold in log intensity.
		cutoff_hz: float,
			3dB cutoff frequency in Hz of DVS photoreceptor
		leak_rate_hz: float
			leak event rate per pixel in Hz,
			from junction leakage in reset switch
		shot_noise_rate_hz: float
			shot noise rate in Hz
		seed: int, default=None
			seed for random threshold variations,
			fix it to integer value to get same mismatch every time
		"""

		self.no_events_warning_count = 0

		self.reset()
		self.t_previous = 0  # time of previous frame
  
		# threshold model
		# pn_related: positive and negative thresholds are related. For each pixel, the mean of the positive and negative thresholds are sampled from a normal distribution with mean thres_mean_mean and std thres_mean_std. The difference between the positive and negative thresholds are sampled from a normal distribution with mean thres_diff_mean and std thres_diff_std.
		# spatial_temporal_independent: positive and negative thresholds are spatially and temporally independent. For each pixel and each frame, the positive and negative thresholds are sampled from a normal distribution with mean thres_mean_mean and std thres_mean_std.
		# spatial_independent: positive and negative thresholds are spatially independent but temporally constant. For each pixel, the positive and negative thresholds are sampled from a normal distribution with mean thres_mean_mean and std thres_mean_std.
		# spatial_independent_temporal_changing: positive and negative thresholds are spatially independent but temporally changing. For each pixel, the positive and negative thresholds are sampled from a normal distribution with mean thres_mean_mean and std thres_mean_std. The thresholds are updated by adding a random value sampled from a normal distribution with mean 0 and std thres_diff_std.
		self.threshold_model = threshold_model

		# thresholds
		self.thres_mean_mean = thres_mean_mean
		self.thres_mean_std = thres_mean_std
		self.thres_diff_mean = thres_diff_mean
		self.thres_diff_std = thres_diff_std
  
		# initialized to scalar, later overwritten by random value array
		self.pos_thres = None
		# initialized to scalar, later overwritten by random value array
		self.neg_thres = None
		# nominal (名义上的) thresholds: these variables save the theoretical threshold values before gaussian sampling
		self.pos_thres_nominal = self.thres_mean_mean + self.thres_diff_mean / 2
		self.neg_thres_nominal = self.thres_mean_mean - self.thres_diff_mean / 2

		# non-idealities
		self.cutoff_hz = cutoff_hz
		self.leak_rate_hz = leak_rate_hz
		self.refractory_period_s = refractory_period_s
		self.shot_noise_rate_hz = shot_noise_rate_hz

		self.leak_jitter_fraction = leak_jitter_fraction
		self.noise_rate_cov_decades = noise_rate_cov_decades

		self.SHOT_NOISE_INTEN_FACTOR = 0.25 # this factor models the slight increase of shot noise with intensity

		# generate jax key for random process
		if seed != None:
			np.random.seed(seed)
			random.seed(seed)


	def _init(self, first_frame_linear):
		"""

		Parameters:
		----------
		first_frame_linear: np.ndarray
			the first frame, used to initialize data structures

		Returns:
			new instance
		-------

		"""
		# base_frame are memorized lin_log pixel values
		self.diff_frame = None

		if self.threshold_model == "pn_related":
			pn_mean = np.random.normal(loc=self.thres_mean_mean, scale=self.thres_mean_std, size=first_frame_linear.shape)
			pn_diff = np.random.normal(loc=self.thres_diff_mean, scale=self.thres_diff_std, size=first_frame_linear.shape)
			self.pos_thres = pn_mean + (pn_diff/2)
			self.neg_thres = pn_mean - (pn_diff/2)
			self.change_pos_neg_thres()
   
		elif self.threshold_model == "spatial_temporal_independent" or self.threshold_model == "spatial_independent" or self.threshold_model == "spatial_independent_temporal_changing":
			self.pos_thres = np.random.normal(loc=self.thres_mean_mean, scale=self.thres_mean_std, size=first_frame_linear.shape)
			self.neg_thres = np.random.normal(loc=self.thres_mean_mean, scale=self.thres_mean_std, size=first_frame_linear.shape)
			self.change_pos_neg_thres()

		# Leak noise is done in subtract_leak_current

		# set noise rate array, it's a log-normal distribution
		self.noise_rate_array = np.random.randn(*first_frame_linear.shape).astype(np.float32)
		self.noise_rate_array = np.exp(math.log(10) * self.noise_rate_cov_decades * self.noise_rate_array)


	# Keep the parameters for reference. The function won't be called.
	def set_dvs_params(self, model: str):
		if model == 'clean':
			self.pos_thres = 0.2
			self.neg_thres = 0.2
			self.sigma_thres = 0.02
			self.cutoff_hz = 0
			self.leak_rate_hz = 0
			self.leak_jitter_fraction = 0
			self.noise_rate_cov_decades = 0
			self.shot_noise_rate_hz = 0  # rate in hz of temporal noise events
			self.refractory_period_s = 0

		elif model == 'noisy':
			self.pos_thres = 0.2
			self.neg_thres = 0.2
			self.sigma_thres = 0.05
			self.cutoff_hz = 30
			self.leak_rate_hz = 0.1
			# rate in hz of temporal noise events
			self.shot_noise_rate_hz = 5.0
			self.refractory_period_s = 0
			self.leak_jitter_fraction = 0.1
			self.noise_rate_cov_decades = 0.1
		else:
			raise ValueError(f'unknown model {model}')

	def reset(self):
		'''resets so that next use will reinitialize the base frame
		'''
		# add names of new states to potentially show with --show_model_states all
		self.new_frame: Optional[np.ndarray] = None # new frame that comes in [height, width]
		self.log_new_frame: Optional[np.ndarray] = None #  [height, width]
		self.lp_log_frame: Optional[np.ndarray] = None  # lowpass stage 0
		self.lp_log_frame: Optional[np.ndarray] = None  # stage 1
		self.c_minus_s_frame: Optional[np.ndarray] = None
		self.base_log_frame: Optional[np.ndarray] = None # memorized log intensities at change detector
		self.diff_frame: Optional[np.ndarray] = None  # [height, width]
		self.frame_counter = 0

	def change_pos_neg_thres(self):
		self.pos_thres = np.clip(self.pos_thres, a_min=0.01, a_max=None)
		self.neg_thres = np.clip(self.neg_thres, a_min=0.01, a_max=None)
		# compute variable for shot-noise
		self.pos_thres_pre_prob = np.divide(
			self.pos_thres_nominal, self.pos_thres)
		self.neg_thres_pre_prob = np.divide(
			self.neg_thres_nominal, self.neg_thres)
	
	def generate_events(self, new_frame, t_frame):
		"""Compute events in new frame.

		Parameters
		----------
		new_frame: np.ndarray
			[height, width], NOTE y is first dimension, like in matlab the column, x is 2nd dimension, i.e. row.
		t_frame: float
			timestamp of new frame in float seconds

		Returns
		-------
		voxel_pos, voxel_neg: np.ndarray
			[height, width] arrays of counts of positive / negative events on each pixel.
		"""
  
		if self.threshold_model == "spatial_temporal_independent":
			# For each new frame, re-generate the random thresholds
			self.pos_thres = np.random.normal(loc=self.thres_mean_mean, scale=self.thres_mean_std, size=new_frame.shape)
			self.neg_thres = np.random.normal(loc=self.thres_mean_mean, scale=self.thres_mean_std, size=new_frame.shape)
			self.change_pos_neg_thres()
   
		elif self.threshold_model == "spatial_independent_temporal_changing":
			self.pos_thres += np.random.normal(loc=0, scale=self.thres_diff_std, size=new_frame.shape)
			self.neg_thres += np.random.normal(loc=0, scale=self.thres_diff_std, size=new_frame.shape)
			self.change_pos_neg_thres()

		# base_frame: the change detector input,
		#              stores memorized brightness values
		# new_frame: the new intensity frame input
		# log_frame: the lowpass filtered brightness values

		# update frame counter
		self.frame_counter += 1

		if t_frame < self.t_previous:
			raise ValueError(
				"this frame time={} must be later than "
				"previous frame time={}".format(t_frame, self.t_previous))

		# compute time difference between this and the previous frame
		delta_time = t_frame - self.t_previous

		self.new_frame = new_frame

		# lin-log mapping, if input is not already float32 log input
		self.log_new_frame = lin_log(self.new_frame)

		inten01 = None  # define for later
		if self.cutoff_hz > 0 or self.shot_noise_rate_hz > 0:  # will use later
			# Time constant of the filter is proportional to
			# the intensity value (with offset to deal with DN=0)
			# limit max time constant to ~1/10 of white intensity level
			# Will be between 0.072 and 1 for 8 bit intensity
			inten01 = rescale_intensity_frame(self.new_frame)
			
		# Apply nonlinear lowpass filter here.
		# Filter is a 1st order lowpass IIR (can be 2nd order)
		# that uses two internal state variables
		# to store stages of cascaded first order RC filters.
		# Time constant of the filter is proportional to
		# the intensity value (with offset to deal with DN=0)
		if self.base_log_frame is None:
			# initialize 1st order IIR to first input
			self.lp_log_frame = self.log_new_frame

		self.lp_log_frame = low_pass_filter(
			log_new_frame=self.log_new_frame,
			lp_log_frame=self.lp_log_frame,
			inten01=inten01,
			delta_time=delta_time,
			cutoff_hz=self.cutoff_hz)
			
		if self.base_log_frame is None:
			self._init(new_frame)
			self.base_log_frame = self.lp_log_frame

			return None  # on first input frame we just setup the state of all internal nodes of pixels

		# Leak events: switch in diff change amp leaks at some rate
		# equivalent to some hz of ON events.
		# Actual leak rate depends on threshold for each pixel.
		# We want nominal rate leak_rate_Hz, so
		# R_l=(dI/dt)/Theta_on, so
		# R_l*Theta_on=dI/dt, so
		# dI=R_l*Theta_on*dt
		if self.leak_rate_hz > 0:
			self.base_log_frame = subtract_leak_current(
				base_log_frame=self.base_log_frame,
				leak_rate_hz=self.leak_rate_hz,
				delta_time=delta_time,
				pos_thres=self.pos_thres,
				leak_jitter_fraction=self.leak_jitter_fraction,
				noise_rate_array=self.noise_rate_array)

		# log intensity (brightness) change from memorized values is computed
		# from the difference between new input
		# (from lowpass of lin-log input) and the memorized value

		# take input from either photoreceptor or amplified high pass nonlinear filtered scidvs
		photoreceptor = self.lp_log_frame

		self.diff_frame = photoreceptor - self.base_log_frame
		
		# generate event map from diff frame
		pos_evts_frame, neg_evts_frame = compute_event_map(
			self.diff_frame, self.pos_thres, self.neg_thres)


		# NOISE: add shot temporal noise here by
		# simple Poisson process that has a base noise rate
		# self.shot_noise_rate_hz.
		# the shot noise rate varies with intensity:
		# for lowest intensity the rate rises to parameter.
		# the noise is reduced by factor
		# SHOT_NOISE_INTEN_FACTOR for brightest intensities
		if self.shot_noise_rate_hz > 0:
			# generate all the noise events for this entire input frame; there could be (but unlikely) several per pixel but only 1 on or off event is returned here
			pos_shot_noise, neg_shot_noise = generate_shot_noise(
				shot_noise_rate_hz=self.shot_noise_rate_hz,
				delta_time=delta_time,
				shot_noise_inten_factor=self.SHOT_NOISE_INTEN_FACTOR,
				inten01=inten01,
				pos_thres_pre_prob=self.pos_thres_pre_prob,
				neg_thres_pre_prob=self.neg_thres_pre_prob)		
		else:
			pos_shot_noise = np.zeros_like(pos_evts_frame)
			neg_shot_noise = np.zeros_like(neg_evts_frame)

		final_pos_evts_frame = pos_evts_frame + pos_shot_noise
		final_neg_evts_frame = neg_evts_frame + neg_shot_noise

		# If all events are uniformly distributed, then the number of events per pixel is limited by the refractory period. (Actually it is the sum of pos & neg that is limited, but let's hunong this.)
		if self.refractory_period_s > 0:
			max_possible_evs = int(delta_time / self.refractory_period_s)
			final_pos_evts_frame = np.clip(final_pos_evts_frame, a_max=max_possible_evs)
			final_neg_evts_frame = np.clip(final_neg_evts_frame, a_max=max_possible_evs)

		# update base log frame according to the number of signal events
		# update the base frame, after we know how many events per pixel
		# add to memorized brightness values just the events we emitted.
		# don't add the remainder.
		# the next aps frame might have sufficient value to trigger
		# another event, or it might not, but we are correct in not storing
		# the current frame brightness
		# Assume that the shot noise events won't effect the base log frame. It is said that the base_log_frame would be set to the correct value of the current frame when shot noise is triggered (?) so ideality will be better if we consider shot noise (which we don't do).
		self.base_log_frame += final_pos_evts_frame * self.pos_thres
		self.base_log_frame -= final_neg_evts_frame * self.neg_thres

		# assign new time
		self.t_previous = t_frame

		return final_pos_evts_frame, final_neg_evts_frame


def video_to_voxel(video, FPS, threshold_model, thres_mean_mean, thres_mean_std, thres_diff_mean, thres_diff_std, cutoff_hz, leak_rate_hz, refractory_period_s, shot_noise_rate_hz, leak_jitter_fraction, noise_rate_cov_decades, seed):
	'''
	video: np.array, [num_frames, height, width]
	'''
	emulator = EventEmulator(
		threshold_model=threshold_model,
		thres_mean_mean=thres_mean_mean,
		thres_mean_std=thres_mean_std,
		thres_diff_mean=thres_diff_mean,
		thres_diff_std=thres_diff_std,
		cutoff_hz=cutoff_hz,
		leak_rate_hz=leak_rate_hz,
		refractory_period_s=refractory_period_s,
		shot_noise_rate_hz=shot_noise_rate_hz,
		leak_jitter_fraction=leak_jitter_fraction,
		noise_rate_cov_decades=noise_rate_cov_decades,
		seed=seed,
	)
	frame_cnt = video.shape[0]
	voxel_frames = []
	for i in range(frame_cnt):
		out = emulator.generate_events(video[i], i/FPS)
		if i > 0:
			pos_voxel, neg_voxel = out
			voxel_frames.append((pos_voxel - neg_voxel))
	return np.array(voxel_frames)

if __name__ == "__main__":
	video = cv2.VideoCapture("/mnt/ssd/Adobe240FPS_gray/GOPR9633.mp4")
	fps = video.get(cv2.CAP_PROP_FPS)
	print("FPS: ", fps)
	start_frame = 100
	end_frame = 201
	height = 300
	width = 300
	vid = np.zeros((end_frame-start_frame, height, width))
	video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
	for i in range(end_frame-start_frame):
		ret, frame = video.read()
		img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		vid[i] = img[:height, :width]

	
	noisy_voxel = video_to_voxel(
		vid, FPS=fps, pos_thres=0.2, neg_thres=0.2, sigma_thres=0.05, cutoff_hz=30, leak_rate_hz=0.1, refractory_period_s=0, shot_noise_rate_hz=5.0, leak_jitter_fraction=0.1, noise_rate_cov_decades=0.1, seed=0)
	
	for i in range(0, noisy_voxel.shape[0], 5):
		viz_noisy_voxel = viz_voxel(np.sum(noisy_voxel[i:i+5], axis=0))
		cv2.imwrite(f"debug/noisy_voxel/{i:03d}.png", viz_noisy_voxel)
	

	clean_voxel = video_to_voxel(
		vid, FPS=fps, pos_thres=0.2, neg_thres=0.2, sigma_thres=0.02, cutoff_hz=0, leak_rate_hz=0, refractory_period_s=0, shot_noise_rate_hz=0, leak_jitter_fraction=0, noise_rate_cov_decades=0, seed=0)
	
	for i in range(0, clean_voxel.shape[0], 5):
		viz_clean_voxel = viz_voxel(np.sum(clean_voxel[i:i+5], axis=0))
		cv2.imwrite(f"debug/clean_voxel/{i:03d}.png", viz_clean_voxel)
		cv2.imwrite(f"debug/vid/{i:03d}.png", vid[i])
