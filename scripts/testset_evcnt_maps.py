# Produce visualizations for Figure 10 of the V2V paper.

import tqdm
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

def read_txt(filename):
	files = []
	with open(filename, "r") as f:
		for line in f:
			files.append(line.strip())
	return files

def make_plot(h5file, out_path):
	with h5py.File(h5file, "r") as f:
		img_keys = sorted(f["images"].keys())
		H, W = f["images"][img_keys[0]].shape
		evcnt = np.zeros((H, W))
		xs = f["events/xs"][:]
		ys = f["events/ys"][:]
		np.add.at(evcnt, (ys, xs), 1)
		# Use log scale for visualization but show actual values
		evcnt_log = np.log1p(evcnt)  # log(1+x) handles zeros gracefully
		im = plt.imshow(evcnt_log, cmap="jet", vmin=0, vmax=np.max(evcnt_log))
		ax = plt.gca()
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.05)
		cbar = plt.colorbar(im, cax=cax)

		# Create custom ticks that show original values
		max_val = evcnt.max()
		if max_val > 10000:
			ticks = [1, 10, 100, 1000, 10000]
		if max_val > 1000:
			ticks = [1, 10, 100, 1000]
		elif max_val > 100:
			ticks = [1, 10, 100]
		else:
			ticks = [1, 10]
		ticks = [t for t in ticks if t <= max_val]
		log_ticks = [np.log1p(t) for t in ticks]  # Convert to log scale
		cbar.set_ticks(log_ticks)
		cbar.set_ticklabels([str(t) for t in ticks])  # Show original values
		cbar.ax.tick_params(labelsize=8)
		ax.axis("off")
		# Add a dummy label in white font with number 10000, so all images have the same width
		ax.text(0.5, 0.5, "100000000", color="white", fontsize=8, ha="left", va="top", transform=cax.transAxes)

		plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=300)
		plt.close()

for classname, filepath in [
	("MVSEC", "config/mvsec_test.txt"),
	("IJRR", "config/ijrr_test.txt"),
	("HQF", "config/hqf_test.txt"),
	("EVAID", "config/evaid_test.txt"),
]:
	files = read_txt(filepath)
	for file in tqdm.tqdm(files):
		seqname = file.split("/")[-1].split(".")[0]
		out_path = f"videos/event_count_maps/{classname}/{seqname}.png"
		os.makedirs(os.path.dirname(out_path), exist_ok=True)
		make_plot(file, out_path)
