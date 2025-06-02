import numpy as np

def reverse_gamma_correction(imgs, gamma=2.2):
	return (imgs / 255) ** gamma * 255

class EventEmulator(object):

	def __init__(
			self,
			pos_thres: float = 0.2,
			neg_thres: float = 0.2,
			base_noise_std: float = 0.1,
			hot_pixel_fraction: float = 0.001,
			hot_pixel_std: float = 0.1,
			put_noise_external: bool = False,
			seed: int = None,
	):
		self.pos_threshold = pos_thres
		self.neg_threshold = neg_thres
		self.base_noise_std = base_noise_std
		self.hot_pixel_fraction = hot_pixel_fraction
		self.hot_pixel_std = hot_pixel_std
		self.put_noise_external = put_noise_external
		self.seed = seed

	def video_to_voxel(self, video):
		N, H, W = video.shape
		# Initialize the potential uniform random between -neg_thres and pos_thres
		self.potential = np.random.rand(H, W) * (self.pos_threshold + self.neg_threshold) - self.neg_threshold

		all_voxels = []
		# Reverse gamma correction will make video more linear.
		video = reverse_gamma_correction(video)
		log_imgs = np.log(0.001 + video/255.0)

		# The hot noise persists for the entire video
		hot_pixel_mask = np.random.rand(H, W) < self.hot_pixel_fraction
		hot_noise = self.hot_pixel_std * np.random.randn(H, W)
		hot_noise = np.where(hot_pixel_mask, hot_noise, 0)		
		
		for i in range(N-1):
			diff = log_imgs[i+1] - log_imgs[i]
			self.potential += diff
			base_noise = self.base_noise_std * np.random.randn(H, W)
			
			if not self.put_noise_external:
				# The noise influences the potential.
				self.potential += base_noise
				self.potential += hot_noise

			pos_events = np.floor_divide(self.potential, self.pos_threshold)
			pos_events = np.where(self.potential >= self.pos_threshold, pos_events, 0)
			
			neg_events = np.floor_divide(-self.potential, self.neg_threshold)
			neg_events = np.where(self.potential <= -self.neg_threshold, neg_events, 0)

			self.potential -= pos_events * self.pos_threshold
			self.potential += neg_events * self.neg_threshold

			voxel = pos_events - neg_events
			
			if self.put_noise_external:
				# Directly add the noise (a float) to the voxel output
				voxel = voxel + base_noise
				voxel = voxel + hot_noise

			all_voxels.append(voxel)
		
		return np.array(all_voxels)
