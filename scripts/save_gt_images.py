# This tool script saves the ground truth images & event visualizations of the test datasets for aligned comparison (such as for making videos with scripts/make_ref_videos.py).

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import yaml
import torch
import tqdm
import cv2
import numpy as np
from collections import defaultdict
from utils.data import data_sources

from data.data_interface import make_concat_multi_dataset
from torch.utils.data import DataLoader

def map_color(val, clip=10):
	BLUE = np.expand_dims(np.expand_dims(np.array([255, 0, 0]), 0), 0)
	RED = np.expand_dims(np.expand_dims(np.array([0, 0, 255]), 0), 0)
	WHITE = np.expand_dims(np.expand_dims(np.array([255, 255, 255]), 0), 0)
	val = np.clip(val, -clip, clip)
	val = np.expand_dims(val, -1)
	red_side = (1 - val / clip) * WHITE + (val / clip) * RED
	blue_side = (1 + val / clip) * WHITE + (-val / clip) * BLUE
	return np.where(val > 0, red_side, blue_side).astype(np.uint8)


def create_test_dataloader(stage_cfg):
	dataset = make_concat_multi_dataset(stage_cfg["test"])
	dataloader = DataLoader(dataset,
							batch_size=1,
							num_workers=stage_cfg["test_num_workers"],
							shuffle=False)
	return dataloader

def save_gt(dataloader, output_dir):

	previous_test_sequence = None

	with torch.no_grad():

		for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
			sequence_name = batch["sequence_name"][0][0]
			
			if previous_test_sequence is None or previous_test_sequence != sequence_name:
				output_img_idx = 0
				if output_dir is not None:
					data_source_idx = batch["data_source_idx"][0]
					data_source = data_sources[data_source_idx].upper()
					seq_output_dir = os.path.join(output_dir, data_source, sequence_name)
					#print("seq_output_dir:", seq_output_dir)
					os.makedirs(seq_output_dir, exist_ok=True)
			
			pred = batch["frame"]

			if output_dir is not None:
				one, T, one, H, W = pred.shape
				for t in range(T):
					img = pred[0, t, 0].cpu().numpy()
					img = np.clip(img, 0, 255).astype(np.uint8)
					cv2.imwrite(os.path.join(seq_output_dir, f"{output_img_idx:06d}.png"), img)
					output_img_idx += 1

			previous_test_sequence = sequence_name

def save_evs(dataloader, output_dir):

	previous_test_sequence = None

	with torch.no_grad():

		for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
			sequence_name = batch["sequence_name"][0][0]
			
			if previous_test_sequence is None or previous_test_sequence != sequence_name:
				output_img_idx = 0
				if output_dir is not None:
					data_source_idx = batch["data_source_idx"][0]
					data_source = data_sources[data_source_idx].upper()
					seq_output_dir = os.path.join(output_dir, data_source, sequence_name)
					#print("seq_output_dir:", seq_output_dir)
					os.makedirs(seq_output_dir, exist_ok=True)
			
			voxel = batch["events"]
			one, T, B, H, W = voxel.shape
			for t in range(T):
				vis = map_color(voxel[0, t, :, :, :].sum(axis=0).cpu().numpy(), clip=5)
				
				cv2.imwrite(os.path.join(seq_output_dir, f"{output_img_idx:06d}.png"), vis)
				output_img_idx += 1

			previous_test_sequence = sequence_name


def main(output_dir, output_evs_dir):
	# Add two arguments.
	# Argument 1: config_path
	# Argument 2 (optional): test_all_pths (default=False)
	if len(sys.argv) > 1:
		config_path = sys.argv[1]
	else:
		config_path = "configs/template.yaml"

	with open(config_path) as f:
		config = yaml.load(f, Loader=yaml.Loader)

	test_dataloader = create_test_dataloader(config["test_stage"])
	save_evs(test_dataloader, output_evs_dir)
	save_gt(test_dataloader, output_dir)
	
if __name__ == "__main__":
	main("results/gt_images", "results/gt_evs")