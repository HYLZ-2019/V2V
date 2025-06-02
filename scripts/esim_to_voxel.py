# Code used to convert h5 ESIM events to cached h5 voxels.

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import DynamicH5Dataset
import glob
import h5py
import numpy as np
import tqdm

original_paths = sorted(glob.glob("/mnt/ssd/esim_h5/*.h5"))
os.makedirs("/mnt/ssd/esim_voxel_nobi", exist_ok=True)
os.makedirs("/mnt/ssd/esim_voxel_cache", exist_ok=True)

def convert(temporal_bilinear, output_file_name):
	for p in tqdm.tqdm(original_paths):
		out_path = p.replace("esim_h5", output_file_name)
		print("Saving to ", out_path)
		dataset = DynamicH5Dataset(data_path=p, temporal_bilinear=temporal_bilinear)
		
		all_frames = []
		all_flow = []
		all_events = []
		all_timestamps = []
		all_dt = []
		
		for i in range(len(dataset)):
			item = dataset[i]
			all_frames.append(item["frame"].numpy())
			all_flow.append(item["flow"].numpy())
			all_events.append(item["events"].numpy())
			all_timestamps.append(item["timestamp"])
			all_dt.append(item["dt"])

		with h5py.File(out_path, "w") as f:
			
			all_frames = np.stack(all_frames)
			all_flow = np.stack(all_flow)
			all_events = np.stack(all_events)
			all_timestamps = np.stack(all_timestamps)
			all_dt = np.stack(all_dt)
			
			f.attrs["sensor_resolution"] = dataset.sensor_resolution
			f.attrs["source"] = "esim"
			f.create_dataset(f"frames", data=all_frames, dtype=np.float32)
			f.create_dataset(f"flow", data=all_flow, dtype=np.float32)
			f.create_dataset(f"events", data=all_events, dtype=np.float32)
			f.create_dataset(f"timestamps", data=all_timestamps, dtype=np.float32)
			f.create_dataset(f"dt", data=all_dt, dtype=np.float32)
			print(all_frames.shape)
			print(all_flow.shape)
			print(all_events.shape)
			print(all_timestamps.shape)
			print(all_dt.shape)
	
convert(True, "esim_voxel_cache") # Already done with previous code, not tested with this code yet
convert(False, "esim_voxel_nobi")