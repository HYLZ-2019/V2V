# Convert HS-ERGB dataset to h5 format.
# We didn't use the dataset for evaluation because (1) the backgrounds were too static, and (2) the GT images were too noisy.
# HS-ERGB download site: https://rpg.ifi.uzh.ch/TimeLens.html

import os
import h5py
import numpy as np
import glob
import cv2

def convert(evaid_dir, h5_path):
	'''
	Download the dataset from our project page. The dataset structure is as follows
	.
	├── close
	│   └── test
	│       ├── baloon_popping
	│       │   ├── events_aligned
	│       │   └── images_corrected
	│       ├── candle
	│       │   ├── events_aligned
	│       │   └── images_corrected
	│       ...
	│
	└── far
		└── test
			├── bridge_lake_01
			│   ├── events_aligned
			│   └── images_corrected
			├── bridge_lake_03
			│   ├── events_aligned
			│   └── images_corrected
			...

	Each events_aligned folder contains events files with template filename %06d.npz, and images_corrected contains image files with template filename %06d.png. In events_aligned each event file with index n contains events between images with index n-1 and n, i.e. event file 000001.npz contains events between images 000000.png and 000001.png. Each event file contains keys for the x,y,t, and p event component. Note that x and y need to be divided by 32 before use. This is because they actually correspond to remapped events, which have floating point coordinates.

	Moreover, images_corrected also contains timestamp.txt where image timestamps are stored. Note that in some folders there are more image files than event files. However, the image stamps in timestamp.txt should match with the event files and the additional images can be ignored.
	'''

	of = h5py.File(h5_path, 'w')

	# Read timestamps
	timestamps_path = os.path.join(evaid_dir, 'images/timestamp.txt')
	with open(timestamps_path, 'r') as f:
		timestamps = f.readlines()
	timestamps = [float(x.strip()) for x in timestamps] # The timestamps are integers e.g. 2810536.0

	all_img_paths = sorted(glob.glob(os.path.join(evaid_dir, "images/*.png")))

	# Read shape from first image
	img0 = cv2.imread(all_img_paths[0])
	H, W, C = img0.shape
	print("H, W, C: ", H, W, C)
	of.create_dataset('sensor_resolution', data=[H, W])

	# Read events
	all_xs = []
	all_ys = []
	all_ts = []
	all_ps = []

	ev_paths = sorted(glob.glob(os.path.join(evaid_dir, 'events/*.npz')))
	for evp in ev_paths:
		# Use accelerated reading with pandas
		ev = np.load(evp)
		xs = ev['x'] // 32 # Throw away the floating point parts, leave this shit to later
		ys = ev['y'] // 32
		ts = ev["timestamp"]
		ps = ev["polarity"]

		# Filter out all events with xs >= W and ys >= H
		mask = np.logical_and(xs < W, ys < H)
		xs = xs[mask]
		ys = ys[mask]
		ts = ts[mask]
		ps = ps[mask]

		if xs.shape[0] > 0:
			all_xs.append(xs)
			all_ys.append(ys)
			all_ps.append(ps)
			all_ts.append(ts)

	all_xs = np.concatenate(all_xs)
	all_ys = np.concatenate(all_ys)
	all_ts = np.concatenate(all_ts)
	all_ps = np.concatenate(all_ps)

	event_idxs = np.searchsorted(all_ts, timestamps)
	basetime = all_ts[0]
	all_ts = (all_ts - basetime).astype(np.float64) / 1e6
	timestamps = (np.array(timestamps) - basetime).astype(np.float64) / 1e6

	of.create_dataset('events/ts', data=ts, dtype=np.float32)
	of.create_dataset('events/xs', data=xs, dtype=np.int16)
	of.create_dataset('events/ys', data=ys, dtype=np.int16)
	of.create_dataset('events/ps', data=ps, dtype=np.bool_)

	frame_cnt = min(len(timestamps), len(all_img_paths))

	for i in range(frame_cnt):
		img = cv2.imread(all_img_paths[i], cv2.IMREAD_GRAYSCALE)
		of.create_dataset(f'images/{i:06d}', data=img)
		of["images"][f'{i:06d}'].attrs['timestamp'] = timestamps[i]
		of["images"][f'{i:06d}'].attrs['event_idx'] = event_idxs[i]
	of.close()	

def process(seqname):
	evaid_dir = f"/mnt/ssd/bs_ergb/1_TEST/{seqname}"
	h5_path = f"/mnt/ssd/bs_ergb/test/{seqname}.h5"
	os.makedirs(os.path.dirname(h5_path), exist_ok=True)
	convert(evaid_dir, h5_path)


all_sequences = [
	os.path.basename(x) for x in glob.glob("/mnt/ssd/bs_ergb/1_TEST/*")
]
print(sorted(all_sequences))
for seqname in sorted(all_sequences):
	print(seqname)
	process(seqname)