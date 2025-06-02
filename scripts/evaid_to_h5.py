# Download EVAID-R from: https://sites.google.com/view/eventaid-benchmark/home

import os
import h5py
import numpy as np
import glob
import pandas as pd
import cv2
import subprocess

def convert(evaid_dir, h5_path, begin_second, end_second):
	# Evaid format:
	# event/*.txt: Events. Each txt is a series of events, each events takes a line: "{timestamp}, {x}, {y}, {polarity}", e.g. "4775805 1131 644 0".
	# gt/*.png: Images. Each png is a frame.
	# shape.txt: Txt with a single line "{W} {H}".
	# timestamps.txt: The i-th line corresponds to the timestamp of the i-th image, e.g. "4775787".
	# event/000001.txt are the events between gt/000001_img.png and gt/000002_img.png. There are no events before the first image, so when converting to h5 we will discard the first image.
	of = h5py.File(h5_path, 'w')

	all_events = []

	# Read timestamps
	timestamps_path = os.path.join(evaid_dir, 'timestamps.txt')
	with open(timestamps_path, 'r') as f:
		timestamps = f.readlines()
	timestamps = [int(x.strip()) for x in timestamps]

	timestamp_rel = np.array(timestamps) - timestamps[0]
	begin_idx = np.searchsorted(timestamp_rel, begin_second * 1e6)
	end_idx = np.searchsorted(timestamp_rel, end_second * 1e6)
	print("begin_idx", begin_idx)
	print("end_idx", end_idx)
	timestamps = timestamps[begin_idx:end_idx+1]
	
	image_paths = sorted(glob.glob(os.path.join(evaid_dir, 'gt/*.png'))) + sorted(glob.glob(os.path.join(evaid_dir, 'gt/*.jpg')))
	image_paths = image_paths[begin_idx:end_idx+1]

	# Read shape
	shape_path = os.path.join(evaid_dir, 'shape.txt')
	with open(shape_path, 'r') as f:
		shape = f.readlines()[0].strip().split(' ')
	W = int(shape[0])
	H = int(shape[1])
	of.create_dataset('sensor_resolution', data=[H, W])

	# Read events
	ev_paths = sorted(glob.glob(os.path.join(evaid_dir, 'event/*.txt')))[begin_idx:end_idx+2]
	for evp in ev_paths:
		# Use accelerated reading with pandas
		ev = pd.read_csv(evp, header=None, sep=' ', names=['timestamp', 'x', 'y', 'polarity'])
		ev = ev.to_numpy()
		if ev.shape[0] > 0:
			all_events.append(ev)
	all_events = np.concatenate(all_events)
	
	ts = all_events[:, 0]
	xs = all_events[:, 1]
	print("xs.shape", xs.shape)
	ys = all_events[:, 2]
	ps = all_events[:, 3]

	event_idxs = np.searchsorted(ts, timestamps)
	basetime = ts[0]
	ts = (ts - basetime).astype(np.float64) / 1e6
	timestamps = (np.array(timestamps) - basetime).astype(np.float64) / 1e6

	of.create_dataset('events/ts', data=ts, dtype=np.float32)
	of.create_dataset('events/xs', data=xs, dtype=np.int16)
	of.create_dataset('events/ys', data=ys, dtype=np.int16)
	of.create_dataset('events/ps', data=ps, dtype=np.bool_)


	
	for i, imgp in enumerate(image_paths):
		if i == 0: # Discard first image
			continue
		img = cv2.imread(imgp, cv2.IMREAD_GRAYSCALE)
		of.create_dataset(f'images/{i:06d}', data=img)
		of["images"][f'{i:06d}'].attrs['timestamp'] = timestamps[i]
		of["images"][f'{i:06d}'].attrs['event_idx'] = event_idxs[i]
	of.close()	

def process(seqname, begin_second, end_second):
	# First, for {seqname}, excecute:
	# unzip /mnt/ssd/EventAid-R/{seqname}.zip -d /mnt/ssd/EventAid-R/{seqname}
	# Check if directory already exists to avoid re-extraction
	if not os.path.exists(f"/mnt/ssd/EventAid-R/{seqname}"):
		subprocess.run(["unzip", f"/mnt/ssd/EventAid-R/{seqname}.zip", "-d", f"/mnt/ssd/EventAid-R/{seqname}"], check=True)
	else:
		print(f"Directory for {seqname} already exists, skipping extraction")

	evaid_dir = f"/mnt/ssd/EventAid-R/{seqname}"
	h5_path = f"/mnt/ssd/EventAid-R-h5/{seqname}.h5"
	os.makedirs(os.path.dirname(h5_path), exist_ok=True)
	convert(evaid_dir, h5_path, begin_second, end_second)

'''
ball：可以有
bear：可以有
blocks：不行，背景完全没有
box：可以有
building：可以有
outdoor：可以有
playball：往后剪
room1：可以有
room2：和room1重复了
scuplture：可以有
toy：可以有
traffic：可以有
umbrella：不行，背景完全没有
wall：可以有
'''

use_seqs = {
	"ball": [0, 5],
	"bear": [0, 5],
	"box": [0, 5],
	"building": [0, 5],
	"outdoor": [0, 5],
	"playball": [25, 30],
	"room1": [0, 5],
	"sculpture": [0, 5],
	"toy": [0, 5],
	"traffic": [0, 5],
	"wall": [0, 5]
}

for seqname, (begin_second, end_second) in use_seqs.items():
	process(seqname, begin_second, end_second)