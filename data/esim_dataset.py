import torch
import h5py
import random
import numpy as np
from utils.data import data_sources

def add_hot_pixels_to_voxels(voxels, hot_pixel_std=1.0, max_hot_pixel_fraction=0.001, integer_noise=False):
	# voxels.shape = (T, C, H, W)
	T, C, H, W = voxels.shape
	hot_pixel_fraction = random.uniform(0, max_hot_pixel_fraction)
	num_hot_pixels = int(hot_pixel_fraction * H * W)
	x = np.random.randint(0, W, num_hot_pixels)
	y = np.random.randint(0, H, num_hot_pixels)
	if integer_noise:
		# Model the noise N = y * sign, where y is a poisson distribution with lambda, and sign  is 50% prob +1, 50% prob -1.
		# Then var(N) will have variance lambda**2 + lambda.
		# We hope lambda**2 + lambda == hot_pixel_std**2.
		# So lambda = (-1 + sqrt(1 + 4s**2)) / 2.
		lmb = (-1 + np.sqrt(1 + 4 * hot_pixel_std**2)) / 2
		y = np.random.poisson(lam=lmb, size=num_hot_pixels)
		sign = 2 * np.random.randint(0, 2, size=num_hot_pixels) - 1
		val = y * sign
	else:
		val = np.random.randn(num_hot_pixels)
		val *= hot_pixel_std
	noise = np.zeros((H, W))
	np.add.at(noise, (y, x), val)
	noise = noise[np.newaxis, np.newaxis, ...]
	voxels += noise
	return voxels


def add_noise_to_voxel(voxel, noise_std=1.0, noise_fraction=0.1, integer_noise=False):
	if integer_noise:
		# lambda-poisson * (50% +1, 50% -1)
		lmb = (-1 + np.sqrt(1 + 4 * noise_std**2)) / 2
		y = np.random.poisson(lam=lmb, size=voxel.shape)
		sign = 2 * np.random.randint(0, 2, size=voxel.shape) - 1
		noise = y * sign
	else:
		noise = noise_std * np.random.randn(*voxel.shape)  # mean = 0, std = noise_std
	
	if noise_fraction < 1.0:
		mask = np.random.rand(*voxel.shape) >= noise_fraction
		noise = np.where(mask, 0, noise)
	return voxel + noise


class ESIMH5Dataset(torch.utils.data.Dataset):
	# The original codebase did not provide HDF5Dataset, which is used in the training configuration.
	# This dataset is similar to DynamicH5Dataset, except that it caches the voxels in h5 file.
	"""
	Dataloader for events saved in the Monash University HDF5 events format
	(see https://github.com/TimoStoff/event_utils for code to convert datasets)
	"""
	
	def __init__(self, h5_path, configs):
		self.h5_path = h5_path
		self.sequence_length = configs.get('sequence_length', 40)
		self.step_size = configs.get('step_size', self.sequence_length)
		self.proba_pause_when_running = configs.get('proba_pause_when_running', 0.05)
		self.proba_pause_when_paused = configs.get('proba_pause_when_paused', 0.9)
		self.noise_std = configs.get('noise_std', 0.1)
		self.noise_fraction = configs.get('noise_fraction', 1.0)
		self.hot_pixel_std = configs.get('hot_pixel_std', 0.1)
		self.max_hot_pixel_fraction = configs.get('max_hot_pixel_fraction', 0.001)
		self.random_crop_size = configs.get('random_crop_size', 112)
		self.random_flip = configs.get('random_flip', True)
		self.integer_noise = configs.get('integer_noise', False)

		self.h5_file = h5py.File(h5_path, 'r')
		self.sensor_resolution = self.h5_file.attrs['sensor_resolution'][0:2]
		self.num_frames = self.h5_file['frames'].shape[0]
		self.data_source_name = "esim"
		self.data_source_idx = data_sources.index(self.data_source_name)

		self.samples = []
		for i in range(0, self.num_frames - self.sequence_length, self.step_size):
			self.samples.append((i, i + self.sequence_length))

	def __len__(self):
		return len(self.samples)
		
	def __getitem__(self, index):
		begin_i, end_i = self.samples[index]

		all_frame = self.h5_file["frames"][begin_i:end_i]
		all_frame = all_frame # in [0, 1]
		all_flow = self.h5_file["flow"][begin_i:end_i]
		all_voxel = self.h5_file["events"][begin_i:end_i]

		# Random crop
		T, one, H, W = all_frame.shape
  
		if self.random_crop_size is not None:
			# Random crop[]
			th, tw = self.random_crop_size, self.random_crop_size
			i = random.randint(0, H - th)
			j = random.randint(0, W - tw)
			all_frame = all_frame[:, :, i:i+th, j:j+tw]
			all_flow = all_flow[:, :, i:i+th, j:j+tw]
			all_voxel = all_voxel[:, :, i:i+th, j:j+tw]

		# Random flip
		if self.random_flip and random.random() > 0.5:
			all_frame = np.flip(all_frame, axis=3)
			all_flow = np.flip(all_flow, axis=3)
			all_voxel = np.flip(all_voxel, axis=3)

		# Random pause
		frame = np.zeros_like(all_frame)
		flow = np.zeros_like(all_flow)
		voxel = np.zeros_like(all_voxel)
		timestamp = []

		paused = False
		k = 0
		for t_idx in range(self.sequence_length):
			# decide whether we should make a "pause" at this step
			# the probability of "pause" is conditioned on the previous state (to encourage long sequences)
			u = np.random.rand()
			if paused:
				probability_pause = self.proba_pause_when_paused
			else:
				probability_pause = self.proba_pause_when_running
			paused = (u < probability_pause)
			if t_idx > 0 and paused: # Cannot pause at the first frame
				# add a tensor filled with zeros, paired with the last frame
				# do not increase the counter
				frame[t_idx] = frame[t_idx - 1]
				# Leave the flow and voxel as zeros

			else:
				# normal case: append the next item to the list
				frame[t_idx] = all_frame[k]
				flow[t_idx] = all_flow[k]
				voxel[t_idx] = all_voxel[k]
				k += 1

			# add noise
			voxel[t_idx] = add_noise_to_voxel(voxel[t_idx], self.noise_std, self.noise_fraction, integer_noise=self.integer_noise)

		voxel = add_hot_pixels_to_voxels(voxel, self.hot_pixel_std, self.max_hot_pixel_fraction, integer_noise=self.integer_noise)
	

		item = {
			'frame': torch.Tensor(frame),
			'flow': torch.Tensor(flow),
			'events': torch.Tensor(voxel),
			'data_source_idx': torch.tensor(self.data_source_idx),
		}

		return item