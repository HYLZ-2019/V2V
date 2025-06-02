import torch
import os
import numpy as np
import cv2
from utils.data import data_sources
from data.v2v_core_esim import EventEmulator
import ffmpeg
#import av
import json
import glob

def log_uniform(minval, maxval):
	eps = 1e-3
	logmin = np.log(minval + eps)
	logmax = np.log(maxval + eps)
	logval = np.random.uniform(logmin, logmax)
	return np.exp(logval) - eps

def bgr_to_gray(img_stack):
	# Convert (N, H, W, 3) to (N, H, W)
	gray = np.dot(img_stack[..., :3], [0.5870, 0.1140, 0.2989])
	return gray.astype(np.uint8)

class WebvidDatasetV2(torch.utils.data.Dataset):

	def load_configs(self, configs):
		self.FPS = configs.get("FPS", 30)

		self.L = configs.get("sequence_length", 40)
		step_size = configs.get("step_size", None)
		
		self.proba_pause_when_running = configs.get("proba_pause_when_running", 0.01)
		self.proba_pause_when_paused = configs.get("proba_pause_when_paused", 0.98)

		self.fixed_seed = configs.get("fixed_seed", None)

		self.crop_size = configs.get("crop_size", None)
		self.fixed_crop = configs.get("fixed_crop", False)
		self.random_flip = configs.get("random_flip", True)
		self.num_bins = configs.get("num_bins", 5)
		self.frames_per_bin = configs.get("frames_per_bin", 1)
		self.frames_per_img = self.num_bins * self.frames_per_bin
		self.frames_per_seq = self.num_bins * self.frames_per_bin * self.L
		# Notice: this does not get non-overlapping samples. Each sample only goes 1 / self.frames_per_img sequences forward.
		self.step_size = step_size if step_size is not None else self.frames_per_seq

		self.min_resize_scale = configs.get("min_resize_scale", 0)
		self.max_resize_scale = configs.get("max_resize_scale", 1.3)
		self.max_rotate_degrees = configs.get("max_rotate_degrees", 0)

		self.shake_frames = configs.get("shake_frames", 0)
		self.shake_std = configs.get("shake_std", 0)

		self.threshold_range = configs.get("threshold_range", [0.05, 2])
		self.max_thres_pos_neg_gap = configs.get("max_thres_pos_neg_gap", 1.5)
		self.base_noise_std_range = configs.get("base_noise_std_range", [0, 0.2])
		self.hot_pixel_fraction_range = configs.get("hot_pixel_fraction_range", [0, 0.001])
		self.hot_pixel_std_range = configs.get("hot_pixel_std_range", [0, 0.2])
		self.put_noise_external = configs.get("put_noise_external", False)
		self.scale_noise_strength = configs.get("scale_noise_strength", False)
		self.max_samples_per_shot = configs.get("max_samples_per_shot", 1)
		self.subsample_ratio = configs.get("subsample_ratio", 1)
		
		self.force_hwaccel = configs.get("force_hwaccel", False)
		self.video_reader = configs.get("video_reader", "ffmpeg")
		assert self.video_reader in ["ffmpeg", "opencv"]

		self.keep_top_percentile = configs.get("keep_top_percentile", 0.54)
		self.use_fixed_thresholds = configs.get("use_fixed_thresholds", False)

		self.data_source_name = configs.get("data_source_name", "reds")
		# The index of data_source_name in data_sources
		self.data_source_idx = data_sources.index(self.data_source_name)

		self.color_mode = configs.get("color_mode", "gray")
		assert self.color_mode in ["gray", "gray_in_bgr_out"]

		assert (self.L > 0)
		assert (self.step_size > 0)

		# Output N+1 frames (0, 5, 10, ..., L*5) instead of N (5, 10, ..., L*5). Used to calculate ground truth optical flow in forward_sequence.
		self.output_additional_frame = configs.get("output_additional_frame", False)
		
		# Output N+1 event voxels. Because ERAFT model needs evs[i-1, i] and evs[i, i+1] to calculate flow[i, i+1].
		self.output_additional_evs = configs.get("output_additional_evs", False)
		if self.output_additional_evs:
			self.frames_per_seq += self.frames_per_img

		# Degrade the video qualities for ablation studies, proving that bad data is not good for training.
		self.video_degrade = configs.get("video_degrade", None)
		assert self.video_degrade in [None, "subtitles", "dirtyshotcut", "hdr", "ldr"]
		self.degrade_ratio = configs.get("degrade_ratio", 0)


	def __init__(self, dataset_path, configs):
		self.load_configs(configs)

		self.dataset_path = dataset_path
		self.video_list_file = configs.get("video_list_file")
		with open(self.video_list_file, "r") as f:
			lines = f.readlines()
			data = [line.strip() for line in lines]
			# Each line: {video_subpath} {video_framecount}
			self.video_list = [line.split(" ")[0] for line in data]
			self.video_framecounts = [int(line.split(" ")[1]) for line in data]

			# Only will be used when self.use_fixed_thresholds
			self.video_pos_thres = [float(line.split(" ")[2]) for line in data]
			self.video_neg_thres = [float(line.split(" ")[3]) for line in data]
	
		self.sample_video_name = []
		self.sample_begin_idx = []
		self.sample_L = []
		self.sample_pos_thres = []
		self.sample_neg_thres = []

		for video_idx, (video_pth, frame_cnt) in enumerate(zip(self.video_list, self.video_framecounts)):
			shot_samples = 0
			for i in range(0, frame_cnt-self.frames_per_seq-1, self.step_size):
				self.sample_video_name.append(video_pth)
				self.sample_begin_idx.append(i)
				self.sample_L.append(self.L)
				self.sample_pos_thres.append(self.video_pos_thres[video_idx])
				self.sample_neg_thres.append(self.video_neg_thres[video_idx])
				shot_samples += 1
				if shot_samples >= self.max_samples_per_shot:
					break

		self.sample_video_name = np.array(self.sample_video_name)
		self.sample_begin_idx = np.array(self.sample_begin_idx)
		self.sample_L = np.array(self.sample_L)

		actual_sample_cnt = int(len(self.sample_L) * self.subsample_ratio)
		self.sample_video_name = self.sample_video_name[:actual_sample_cnt]
		self.sample_begin_idx = self.sample_begin_idx[:actual_sample_cnt]
		self.sample_L = self.sample_L[:actual_sample_cnt]
		self.sample_pos_thres = self.sample_pos_thres[:actual_sample_cnt]
		self.sample_neg_thres = self.sample_neg_thres[:actual_sample_cnt]

		#print(len(self.sample_video_name), len(self.video_pos_thres), len(self.video_neg_thres))

	def __len__(self):
		return len(self.sample_video_name)

	def read_video(self, video_path, start_frame, end_frame, crop_size_before_resize, min_i, min_j, flip):
		all_di = [0] * (end_frame - start_frame)
		all_dj = [0] * (end_frame - start_frame)
		if self.shake_frames > 0:
			# The shake ends with speed 0, so that the video is stable at the end.
			vi = 0
			vj = 0
			di = 0
			dj = 0
			for i in range(min(self.shake_frames, end_frame-start_frame)-1, -1, -1):
				vi += int(np.random.normal(0, self.shake_std))
				vj += int(np.random.normal(0, self.shake_std))
				di += vi
				dj += vj
				all_di[i] = di
				all_dj[i] = dj
    
		extra_h = max(all_di) - min(all_di)
		extra_w = max(all_dj) - min(all_dj)
		need_h = self.crop_size + extra_h
		need_w = self.crop_size + extra_w

		if self.video_reader == "ffmpeg":
			assert False, "FFMPEG hasn't been updated to support color."
			if self.force_hwaccel:
				pipeline = ffmpeg.input(video_path, hwaccel='cuda')
			else:
				pipeline = ffmpeg.input(video_path)

			pipeline = pipeline\
				.filter('trim', start_frame=start_frame, end_frame=end_frame)\
				.filter('crop', w=crop_size_before_resize, h=crop_size_before_resize, x=min_j, y=min_i)\
				.filter('scale', w=need_w, h=need_h)
			if flip:
				pipeline = pipeline.hflip()
			pipeline = pipeline.output('pipe:', format='rawvideo', pix_fmt='gray', loglevel='quiet')

			all_patches, stderr = pipeline.run(capture_stdout=True, capture_stderr=True)
			if stderr:
				print("FFMPEG stderr:", stderr.decode())
			imgs = np.frombuffer(all_patches, dtype=np.uint8).reshape(-1, need_h, need_w)

		elif self.video_reader == "opencv":
			cap = cv2.VideoCapture(video_path)
			cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
			imgs = []
			for i in range(start_frame, end_frame):
				ret, frame = cap.read()
				if not ret:
					break

				if self.color_mode == "gray":
					frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				elif self.color_mode == "gray_in_bgr_out":
					# Should be (H, W, 3)
					pass
				
				frame = frame[min_i:min_i+crop_size_before_resize, min_j:min_j+crop_size_before_resize, ...]
				frame = cv2.resize(frame, (need_w, need_h), interpolation=cv2.INTER_LINEAR)

				if flip:
					frame = cv2.flip(frame, 1)
				
				if self.color_mode == "gray":
					# Expand dimensions to (H, W, 1) for consistency with RGB
					# expand_dims needs to be done after cv2.resize and cv2.flip, because cv2.resize will eat the (,1) dimension
					frame = np.expand_dims(frame, axis=-1)
				
				imgs.append(frame)

			cap.release()
		
		# Shake by cropping
		all_di = np.array(all_di) - min(all_di)
		all_dj = np.array(all_dj) - min(all_dj)
		shaked_imgs = []
		for i in range(len(imgs)):
			img = imgs[i]
			img = img[all_di[i]:all_di[i]+self.crop_size, all_dj[i]:all_dj[i]+self.crop_size, :]
			shaked_imgs.append(img)
		return shaked_imgs

	def __getitem__(self, sample_idx):
		""" Returns a list containing synchronized events <-> frame pairs
			[e_{i-L} <-> I_{i-L},
			e_{i-L+1} <-> I_{i-L+1},
			...,
			e_{i-1} <-> I_{i-1},
			e_i <-> I_i]
		"""
		if self.fixed_seed is not None:
			# Fix the random seed, so that the randomness only depends on the idx of the validation batch.
			# Save the previous seed and recover it at the end of this function, so that the randomness of training is not affected.
			old_random_state = np.random.get_state()
			np.random.seed(self.fixed_seed + idx)

		video_name = self.sample_video_name[sample_idx]
		start_frame = self.sample_begin_idx[sample_idx]
		img_cnt = self.sample_L[sample_idx]

		video_path = os.path.join(self.dataset_path, video_name)

		if self.video_reader == "ffmpeg":
			# Get video information
			probe = ffmpeg.probe(video_path)
			video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
			vid_width = int(video_info['width'])
			vid_height = int(video_info['height']) # Only keep top 54% since the shutterstock watermark is in the lower-middle.
		elif self.video_reader == "opencv":
			cap = cv2.VideoCapture(video_path)
			vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			cap.release()
		

		if self.crop_size is not None:
			min_resize_scale = max(
				self.min_resize_scale,
				self.crop_size / int(vid_height * self.keep_top_percentile),
				self.crop_size / vid_width
			)
			max_resize_scale = max(self.max_resize_scale, min_resize_scale)
		else:
			raise NotImplementedError("crop_size must be provided for WebvidDataset.")
	
		resize_scale = np.random.uniform(min_resize_scale, max_resize_scale)
		crop_size_before_resize = int(self.crop_size / resize_scale)

		if self.fixed_crop:
			min_i = 0
			min_j = 0
		else:
			min_i = np.random.randint(0, int(vid_height*self.keep_top_percentile) - crop_size_before_resize + 1)
			min_j = np.random.randint(0, vid_width - crop_size_before_resize + 1)

		# Initialize flip flag
		flip = False
		if self.random_flip and np.random.rand() > 0.5:
			flip = True

		# Get pause sequence
		img_idxes = []
		idx = 0
		is_pause = False

		# The additional frames are used for generating additional events. output_additional_frame does not need them.
		additional_frames = self.frames_per_img if self.output_additional_evs else 0

		for _ in range(start_frame, start_frame + img_cnt * self.frames_per_img + 1 + additional_frames):
			img_idxes.append(idx)
			if is_pause and np.random.rand() > self.proba_pause_when_paused:
				is_pause = False
			elif not is_pause and np.random.rand() < self.proba_pause_when_running:
				is_pause = True
			if not is_pause:
				idx += 1
		true_img_cnt = idx + 1

		end_frame = start_frame + true_img_cnt
		# raw_imgs: list of (H, W, 3) or (H, W, 1) images.
		raw_imgs = self.read_video(video_path, start_frame, end_frame, crop_size_before_resize, min_i, min_j, flip)

		if self.video_degrade is not None and np.random.rand() < self.degrade_ratio:
			raw_imgs = self.degrade_video(raw_imgs)

		# all_imgs: (N, H, W, 3) or (N, H, W, 1) images.
		all_imgs = np.stack([raw_imgs[i] for i in img_idxes])

		if self.color_mode == "gray":
			gray_imgs = all_imgs[..., 0]  # (N, H, W)
		elif self.color_mode == "gray_in_bgr_out":
			gray_imgs = bgr_to_gray(all_imgs)  # (N, H, W)

		FPS = 24
		# (N, num_bins, H, W)
		pos_thres = self.sample_pos_thres[sample_idx] if self.use_fixed_thresholds else None
		neg_thres = self.sample_neg_thres[sample_idx] if self.use_fixed_thresholds else None
		v2e_params, all_voxels = self.imgs_to_voxels(gray_imgs, self.num_bins, self.frames_per_bin, FPS, pos_thres, neg_thres)
		_ = all_voxels.shape  # Shape information not needed

		if self.output_additional_evs:
			all_imgs = all_imgs[self.frames_per_img:] # The beginning frames are only for the additional event generation.

		# T*1*H*W
		if not self.output_additional_frame:
			all_frames = torch.stack([
				# Convert [N, H, W, C] to [N, C, H, W]
				torch.tensor(all_imgs[(i+1)*self.frames_per_img].copy(), dtype=torch.float32).permute(2, 0, 1) for i in range(img_cnt)
			], axis=0)
		else:
			all_frames = torch.stack([
				# Convert [N, H, W, C] to [N, C, H, W]
				torch.tensor(all_imgs[i*self.frames_per_img].copy(), dtype=torch.float32).permute(2, 0, 1) for i in range(img_cnt+1)
			], axis=0)

		if not self.output_additional_evs:
			# T*bin_cnt*H*W
			all_events = torch.stack([
				torch.tensor(all_voxels[i], dtype=torch.float32) for i in range(img_cnt)
			], axis=0)
		else:
			# T*bin_cnt*H*W
			all_events = torch.stack([
				torch.tensor(all_voxels[i], dtype=torch.float32) for i in range(img_cnt+1)
			], axis=0)

		sequence = {
			"frame": all_frames / 255,
			"events": all_events,
			"data_source_idx": torch.tensor(self.data_source_idx),
			"v2e_params": v2e_params,
		}

		if self.fixed_seed is not None:
			np.random.set_state(old_random_state)

		return sequence
	
	def imgs_to_voxels(self, imgs, num_bins, frames_per_bin, FPS, pos_thres=None, neg_thres=None):
		N, H, W = imgs.shape
		assert (N-1) % (num_bins * frames_per_bin) == 0
		frame_cnt = (N-1) // (num_bins * frames_per_bin)

		if not self.use_fixed_thresholds: # Use random thresholds. use_fixed_thresholds is for ablation training.
			thres_1 = np.random.uniform(*self.threshold_range)
			pos_neg_gap = np.random.uniform(1, self.max_thres_pos_neg_gap)
			thres_2 = thres_1 * pos_neg_gap
			if np.random.rand() > 0.5:
				pos_thres = thres_1
				neg_thres = thres_2
			else:
				pos_thres = thres_2
				neg_thres = thres_1

		base_noise_std = np.random.uniform(*self.base_noise_std_range)
		hot_pixel_fraction = np.random.uniform(*self.hot_pixel_fraction_range)
		hot_pixel_std = np.random.uniform(*self.hot_pixel_std_range)

		if self.scale_noise_strength and not self.put_noise_external:
			# The same base_noise_std should lead to the same amount of pure noise events, independent to the threshold.
			base_noise_std = base_noise_std * pos_thres
			hot_pixel_std = hot_pixel_std * pos_thres

		all_voxels = EventEmulator(
			pos_thres=pos_thres,
			neg_thres=neg_thres,
			base_noise_std=base_noise_std,
			hot_pixel_fraction=hot_pixel_fraction,
			hot_pixel_std=hot_pixel_std,
			put_noise_external=self.put_noise_external,
			seed = None
		).video_to_voxel(imgs)

		# Reshape to (N, num_bins, frames_per_bin, H, W)
		all_voxels = all_voxels.reshape((frame_cnt, num_bins, frames_per_bin, H, W))
		frame_voxels = all_voxels.sum(axis=2)

		v2e_params = {
			"pos_thres": pos_thres,
			"neg_thres": neg_thres,
			"base_noise_std": base_noise_std,
			"hot_pixel_fraction": hot_pixel_fraction,
			"hot_pixel_std": hot_pixel_std,
		}

		return v2e_params, frame_voxels
	

	def degrade_video(self, imgs):
		T = len(imgs)
		# imgs: list of (H, W, 3) or (H, W, 1) images.
		if self.video_degrade == "subtitles":
			# Write random text floating on the entire video. The crops are 128*128.
			font_list = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX]
			font = np.random.choice(font_list)
			font_scale = np.random.uniform(0.5, 1.5)
			# Random BGR color
			color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
			thickness = np.random.randint(1, 3)
			
			# Generate random text (e.g., 5-15 characters)
			text_len = np.random.randint(5, 16)
			text = "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "), size=text_len))

			# Get text size to determine position
			H, W = imgs[0].shape[:2]
			(text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
			
			# Random position (ensure text stays within frame)
			org_x = np.random.randint(0, max(1, W - text_width))
			org_y = np.random.randint(text_height, max(text_height + 1, H - baseline)) # Ensure text baseline is within frame
			org = (org_x, org_y)

			for i in range(T):
				# Need to copy the image if it's read-only (e.g., from numpy array)
				img_copy = imgs[i].copy()
				# If the image is grayscale (H, W, 1), convert to BGR for coloring text
				if img_copy.shape[2] == 1:
					img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
				
				cv2.putText(img_copy, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
				
				# If the original was grayscale, convert back
				if imgs[i].shape[2] == 1:
					imgs[i] = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
				else:
					imgs[i] = img_copy
			return imgs
		
		elif self.video_degrade == "dirtyshotcut":
			# Simulate a dirty shot cut by randomly cutting the video into 2 parts, changing their sequence, and flipping one of them.
			H, W, C = imgs[0].shape
			if T < 3:
				return imgs
			cut_idx = np.random.randint(1, T-1)
			flip_first = np.random.rand() > 0.5
			if flip_first:
				# Flip the first part
				imgs[:cut_idx] = [cv2.flip(img, 1) for img in imgs[:cut_idx]]
			else:
				# Flip the second part
				imgs[cut_idx:] = [cv2.flip(img, 1) for img in imgs[cut_idx:]]
			# cv2.flip will cause the images to lose their (,1) dimension.
			if C == 1:
				imgs = [np.expand_dims(img, axis=-1) if img.ndim == 2 else img for img in imgs]
			imgs = imgs[cut_idx:] + imgs[:cut_idx]
			return imgs
		
		elif self.video_degrade == "hdr":
			scale = np.random.uniform(1, 3)
			for i in range(T):
				imgs[i] = np.clip((imgs[i]-127.5) * scale + 127.5, 0, 255).astype(np.uint8)
			return imgs
		
		elif self.video_degrade == "ldr":
			scale = np.random.uniform(0.3, 1)
			for i in range(T):
				imgs[i] = np.clip((imgs[i]-127.5) * scale + 127.5, 0, 255).astype(np.uint8)
			return imgs
		
		else:
			raise NotImplementedError("Video degrade type not supported.")
