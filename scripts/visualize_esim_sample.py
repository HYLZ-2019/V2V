# Draw visualization of events for Figure 1 of the V2V paper.

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.v2v_core_esim import EventEmulator


# Code for generating 3D visualization of events. By HYLZ, 2025.
def visualize_events_3d(evs, patch_size=256, out_path="", alpha_corr=2):

	evs[:, 0] = (evs[:, 0] - evs[0, 0]) / evs[-1, 0] * 500
	T = evs[-1, 0]
	print(evs)
	print(np.max(evs, axis=0))
	print(T)

	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111, projection='3d')
	
	# Remove axis lines and ticks
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_zticks([])
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_zticklabels([])

	# Disable axis panes (remove grey overlay)
	ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
	ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
	ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

	# Turn off axis lines (spines)
	ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
	ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
	ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

	# Rotate the plot so x is horizontal and y is vertical; t is 30 degrees from the vertical
	
	#ax.view_init(elev=10, azim=230, roll=0, vertical_axis='y')
	ax.view_init(elev=10, azim=250, roll=0, vertical_axis='y')
	ax.set_xlim([0, patch_size])
	ax.set_ylim([0, patch_size])
	ax.set_zlim([0, T])

	ax.invert_xaxis()

	arrow_len = 10

	# Draw edges of the cube
	ax.plot([0, 0], [patch_size, patch_size], [0, T], 'k-', alpha=0.3)
	ax.plot([0, 0], [0, 0], [0, T], 'k-', alpha=0.3)
	ax.plot([patch_size, patch_size], [patch_size, patch_size], [0, T], 'k-', alpha=0.3)
	ax.plot([0, 0], [0, patch_size], [T, T], 'k-', alpha=0.3)
	ax.plot([0, patch_size], [0, 0], [T, T], 'k-', alpha=0.3)
	ax.plot([0, patch_size], [patch_size, patch_size], [T, T], 'k-', alpha=0.3)
	ax.plot([patch_size, patch_size], [0, patch_size], [T, T], 'k-', alpha=0.3)

	# Along the T axis, draw alpha=0.1 rectangular white planes. Draw planes & scatter dots sequentially.
	def scatter_evs(begin_t, end_t):
		begin_idx = np.searchsorted(evs[:, 0], begin_t)
		end_idx = np.searchsorted(evs[:, 0], end_t)
		if begin_idx < end_idx:
			colors = np.array(['red' if p > 0 else 'blue' for p in evs[end_idx:begin_idx:-1, 3]])
			ax.scatter(evs[end_idx:begin_idx:-1, 1], patch_size - 1 - evs[end_idx:begin_idx:-1, 2], evs[end_idx:begin_idx:-1, 0], c=colors, marker='.', alpha=0.05*alpha_corr, s=10)
	
	scatter_evs(evs[0, 0], evs[-1, 0])
	
	# Draw T axis
	ax.plot([patch_size, patch_size], [0, 0], [0, T], 'k-', alpha=1, linewidth=2)
	t_ratio = T / patch_size * 1
	width_ratio = 0.3
	ax.plot([patch_size, patch_size+arrow_len*width_ratio*t_ratio], [0, -arrow_len*width_ratio*t_ratio], [T, T-arrow_len*t_ratio], 'k-', alpha=1, linewidth=2)
	ax.plot([patch_size, patch_size-arrow_len*width_ratio*t_ratio], [0, arrow_len*width_ratio*t_ratio], [T, T-arrow_len*t_ratio], 'k-', alpha=1, linewidth=2)
	# Put text "T axis" at the end of the T axis
	ax.text(patch_size, -arrow_len, T+arrow_len, "t", fontsize=30, color='black')
	
	# Y axis
	ax.plot([0, 0], [0, patch_size], [0, 0], 'k-', alpha=1, linewidth=2)
	ax.plot([0, arrow_len*0.5], [patch_size, patch_size-arrow_len], [0, 0], 'k-', alpha=1, linewidth=2)
	ax.plot([0, -arrow_len*0.5], [patch_size, patch_size-arrow_len], [0, 0], 'k-', alpha=1, linewidth=2)
	ax.text(-arrow_len, patch_size+2*arrow_len, 0, "y", fontsize=30, color='black')
	
	# X axis
	ax.plot([0, patch_size], [0, 0], [0, 0], 'k-', alpha=1, linewidth=2)
	ax.text(patch_size+arrow_len, -2*arrow_len, 0, "x", fontsize=30, color='black')
	ax.plot([patch_size, patch_size-arrow_len], [0, -arrow_len*0.5], [0, 0], 'k-', alpha=1, linewidth=2)
	ax.plot([patch_size, patch_size-arrow_len], [0, arrow_len*0.5], [0, 0], 'k-', alpha=1, linewidth=2)
	
	ax.plot([0, patch_size], [patch_size, patch_size], [0, 0], 'k-', alpha=0.3)
	ax.plot([patch_size, patch_size], [0, patch_size], [0, 0], 'k-', alpha=0.3)

	ax.grid(False)

	plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
	plt.close()

def map_color(val, clip=10):
	BLUE = np.expand_dims(np.expand_dims(np.array([255, 0, 0]), 0), 0)
	RED = np.expand_dims(np.expand_dims(np.array([0, 0, 255]), 0), 0)
	WHITE = np.expand_dims(np.expand_dims(np.array([255, 255, 255]), 0), 0)
	val = np.clip(val, -clip, clip)
	val = np.expand_dims(val, -1)
	red_side = (1 - val / clip) * WHITE + (val / clip) * RED
	blue_side = (1 + val / clip) * WHITE + (-val / clip) * BLUE
	return np.where(val > 0, red_side, blue_side).astype(np.uint8)

def make_voxel(evs, H, W, num_bins=5, interpolate_bins=True):
	voxel = np.zeros((num_bins, H, W))
	ts, xs, ys, ps = evs
	if ts.shape[0] == 0:
		return voxel
	
	# ps of hqf h5 file are in {0, 1}.
	ps = ps.astype(np.int8) * 2 - 1
	ts = ((ts - ts[0]) * 1e6).astype(np.int64)
		
	if not interpolate_bins:
		t_per_bin = (ts[-1] + 0.001) / num_bins
		bin_idx = np.floor(ts / t_per_bin).astype(np.uint8)
		np.add.at(voxel, (bin_idx, ys, xs), ps)
	else:
		# Interpolate the events to the bins.
		dt = ts[-1] - ts[0]
		t_norm = (ts - ts[0]) / (dt + 0.0001) * (num_bins - 1)
		for bi in range(num_bins):
			weights = np.maximum(0, 1.0 - np.abs(t_norm - bi))
			np.add.at(voxel, (bi, ys, xs), weights*ps)
			
	return voxel

if __name__ == "__main__":
	os.makedirs("videos/teaser", exist_ok=True)
	with h5py.File("/mnt/ssd/esim_h5/000000000_out.h5", "r") as f:
		frame_keys = sorted(f["images"].keys())
		img0 = f["images"][frame_keys[0]]
		img1 = f["images"][frame_keys[1]]
		begin_idx = img0.attrs["event_idx"]
		end_idx = img1.attrs["event_idx"]
		xs = f['events/xs'][begin_idx:end_idx]
		ys = f['events/ys'][begin_idx:end_idx]
		ts = f['events/ts'][begin_idx:end_idx]
		ps = f['events/ps'][begin_idx:end_idx]

		evs = np.stack([ts, xs, ys, ps], axis=-1)

		visualize_events_3d(evs, patch_size=256, out_path="videos/teaser/esim_evs.png")

		cv2.imwrite("videos/teaser/esim_img0.png", (img0[()]).astype(np.uint8))

		voxels = make_voxel([ts, xs, ys, ps], 256, 256, num_bins=5, interpolate_bins=True)
		for i in range(voxels.shape[0]):
			voxel = voxels[i]
			voxel_vis = map_color(voxel, clip=1)
			cv2.imwrite(f"videos/teaser/esim_voxel_{i}.png", voxel_vis)
	
		# Read
		# 000201_000250/1018468789.mp4
		# video = cv2.VideoCapture("/mnt/ssd/webvid/000201_000250/1018468789.mp4") 390
		video = cv2.VideoCapture("/mnt/ssd/webvid/000101_000150/1039053524.mp4")
		
		imgs = []
		video.set(cv2.CAP_PROP_POS_FRAMES, 600)
		for i in range(6):
			ret, img = video.read()
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = img[120:248, :128]
			imgs.append(img)
			cv2.imwrite(f"videos/teaser/webvid_img_{i}.png", img)
		video.release()

		imgs = np.stack(imgs, axis=0)
		all_voxels = EventEmulator(
			pos_thres=0.5,
			neg_thres=0.5,
			base_noise_std=0.05,
			hot_pixel_fraction=0.0005,
			hot_pixel_std=5,
			put_noise_external=False,
			seed = None
		).video_to_voxel(imgs)

		for i in range(all_voxels.shape[0]):
			voxel = all_voxels[i]
			voxel_vis = map_color(voxel, clip=1)
			cv2.imwrite(f"videos/teaser/webvid_voxel_{i}.png", voxel_vis)