import os
import h5py
import dv_processing as dv
import numpy as np

# Read from a aedat4 file.
# Only keep data from seconds [begin] to [end].
# Save to TestH5Dataset format.

def convert(aedat4_file, h5_file, begin, end):
	reader = dv.io.MonoCameraRecording(aedat4_file)
	# Run the loop while camera is still connected
	base_time = None
	all_x = []
	all_y = []
	all_t = []
	all_p = []
	while reader.isRunning():
		# Read batch of events
		events = reader.getNextEventBatch()
		if events is not None:
			# Print received packet time range
			evs = events.numpy()
			# print(evs.dtype)
			# {'names': ['timestamp', 'x', 'y', 'polarity'], 'formats': ['<i8', '<i2', '<i2', 'i1'], 'offsets': [0, 8, 10, 12], 'itemsize': 16}
			if base_time is None:
				base_time = evs['timestamp'].min()
			time_min = (evs['timestamp'].min() - base_time) / 1e6
			time_max = (evs['timestamp'].max() - base_time) / 1e6
			if time_max < begin or time_min > end:
				continue

			all_x.append(evs['x'])
			all_y.append(evs['y'])
			all_t.append(evs['timestamp'])
			all_p.append(evs['polarity'])

	reader = dv.io.MonoCameraRecording(aedat4_file)
	# Read the images
	all_imgs = []
	img_timestamps = []
	while reader.isRunning():
		# Read a frame from the camera
		frame = reader.getNextFrame()

		if frame is not None:
			timestamp = (frame.timestamp - base_time) / 1e6
			if timestamp < begin or timestamp > end:
				continue 
			all_imgs.append(frame.image)
			img_timestamps.append(frame.timestamp)

	# Save to h5 file
	all_x = np.concatenate(all_x)
	all_y = np.concatenate(all_y)
	base_t = all_t[0][0]
	all_t = (np.concatenate(all_t) - base_t).astype(np.float64) / 1e6
	print(all_t[0])
	print(all_t[-1])

	all_p = np.concatenate(all_p)

	img_event_idxs = np.searchsorted(all_t, img_timestamps)

	with h5py.File(h5_file, 'w') as f:
		f.create_dataset("events/ts", data=all_t, dtype=np.float32)
		f.create_dataset("events/xs", data=all_x, dtype=np.int16)
		f.create_dataset("events/ys", data=all_y, dtype=np.int16)
		f.create_dataset("events/ps", data=all_p, dtype=np.bool_)

		for i, img in enumerate(all_imgs):
			f.create_dataset(f"images/{i:06d}", data=img)
			# Set attribute f["images"][f"{i:06d}"].attrs["event_idx"] = img_event_idxs[i]
			f["images"][f"{i:06d}"].attrs.create("event_idx", img_event_idxs[i])


src_dir = "/mnt/nas-cp/hylou/Datasets/EvBirding/20250331/raw"
dst_dir = "/mnt/nas-cp/hylou/Datasets/EvBirding/20250331/h5"
convert(f"{src_dir}/maque.aedat4", f"{dst_dir}/maque.h5", 0, 10)