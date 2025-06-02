import os
import sys
import yaml
import torch
import tqdm
import cv2
import numpy as np
from collections import defaultdict
from utils.data import data_sources

from model.train_flow_utils import FlowModelInterface, flow2rgb_np
from train import convert_to_compiled
from test_e2vid import create_test_dataloader

metrics = ["dense_EPE", "dense_1PE", "dense_3PE", "sparse_EPE", "sparse_1PE", "sparse_3PE"]
sequences = {
	#"IJRR": ["boxes_6dof", "calibration", "dynamic_6dof", "office_zigzag", "poster_6dof", "shapes_6dof", "slider_depth"],
	"MVSEC": ["indoor_flying1", "indoor_flying2", "indoor_flying3", "outdoor_day1", "outdoor_day2"],
	#"HQF": ["bike_bay_hdr", "boxes", "desk", "desk_fast", "desk_hand_only", "desk_slow", "engineering_posters", "high_texture_plants", "poster_pillar_1", "poster_pillar_2", "reflective_materials", "slow_and_fast_desk", "slow_hand", "still_life"],
	#"EVAID": ["ball", "bear", "box", "building", "outdoor", "playball", "room1", "sculpture", "toy", "traffic", "wall"]
}
all_metric_names = []
for k, v in sequences.items():
	for seqname in v:
		for m in metrics:
			all_metric_names.append(f"{k}/{seqname}/{m}")

def run_test(model_interface, dataloader, device, configs):

	output_dir = configs["test_output_dir"]
	save_npy = configs.get("save_npy", False)
	save_png = configs.get("save_png", True)
	
	# Actually a flow model, but variable name is still e2vid_model
	model_interface.e2vid_model.eval()
	
	previous_test_sequence = None
	all_metrics = defaultdict(list)

	with torch.no_grad():

		for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
			sequence_name = batch["sequence_name"][0][0]
			
			if previous_test_sequence is None or previous_test_sequence != sequence_name:
				model_interface.e2vid_model.reset_states()
				output_img_idx = 0
				if output_dir is not None:
					data_source_idx = batch["data_source_idx"][0]
					data_source = data_sources[data_source_idx].upper()
					seq_output_dir = os.path.join(output_dir, data_source, sequence_name)
					#print("seq_output_dir:", seq_output_dir)
					os.makedirs(seq_output_dir, exist_ok=True)

			for k, v in batch.items():
				if torch.is_tensor(v):
					batch[k] = v.to(device)
			
			pred = model_interface.forward_sequence(batch, reset_states=False, test=True, val=True) # Reset manually according to sequence name
			# pred: (B, T, 2, H, W) optical flow
			
			#pred = torch.zeros_like(pred)
			metrics = model_interface.compute_metrics(pred, batch)
			for k, v in metrics.items():
				if k in all_metric_names:
					all_metrics[k] += v # v is also a list


			if output_dir is not None:
				one, T, C, H, W = pred.shape
				for t in range(T):
					flow = pred[0, t, :].detach().cpu().numpy()
					# gt_flow = batch["flow"][0, t, :].detach().cpu().numpy()
					# gt_flow = np.where(np.isnan(gt_flow), 0, gt_flow)
					# flow = gt_flow
					#cat_flow = np.concatenate([flow, gt_flow], axis=2)
					if save_npy:
						np.save(os.path.join(seq_output_dir, f"{output_img_idx:06d}.npy"), flow)
					if save_png:
						flow_vis = flow2rgb_np(flow[0, :, :], flow[1, :, :])
						#flow_vis = flow2rgb_np(cat_flow[0, :, :], cat_flow[1, :, :])
						cv2.imwrite(os.path.join(seq_output_dir, f"{output_img_idx:06d}_flow.png"), flow_vis)

					output_img_idx += 1

			previous_test_sequence = sequence_name
	
	output_metric_txt = os.path.join("tensorboard_logs", configs["experiment_name"], "test_metrics.txt")
	with open(output_metric_txt, "w") as f:
		for k, v in all_metrics.items():
			all_metrics[k] = np.mean(v)
			print(f"{k}: {all_metrics[k]}")
			f.write(f"{k}: {all_metrics[k]}\n")
			
	return all_metrics


def main():
	# Add two arguments.
	# Argument 1: config_path
	# Argument 2 (optional): test_all_pths (default=False)
	if len(sys.argv) > 1:
		config_path = sys.argv[1]
	else:
		config_path = "configs/template.yaml"

	if len(sys.argv) > 2:
		test_all_pths = True
	else:
		test_all_pths = False

	with open(config_path) as f:
		config = yaml.load(f, Loader=yaml.Loader)

	assert config.get("task", "e2vid") == "flow", "e2vid should be tested with test_torch.py"

	ckpt_paths_file = f"ckpt_paths/{config['experiment_name']}.txt"
	output_csv = os.path.join("tensorboard_logs", config['experiment_name'], f"all_test_results_new.csv")
	os.makedirs(os.path.dirname(output_csv), exist_ok=True)
	done_checkpoints = []
	if os.path.exists(output_csv):
		with open(output_csv, "r", encoding="utf-8") as f:
			lines = f.readlines()
			for line in lines[1:]:
				ckpt_path = line.split(",")[0]
				done_checkpoints.append(ckpt_path)

	# First row: all the metric names
	# Each row: subpath, metric1, metric2, ....	
	if not os.path.exists(output_csv):
		with open(output_csv, "w", encoding="UTF-8") as f:
			f.write("Checkpoint_path,")
			for key in all_metric_names:
				f.write(f"{key},")
			f.write("\n")

	all_results = []
	if os.path.exists(ckpt_paths_file) and os.path.getsize(ckpt_paths_file) > 0:
		with open(ckpt_paths_file, "r") as f:
			paths = [p.strip() for p in f.readlines() if p.strip()]
			assert len(paths) > 0, "No checkpoint paths found in the file."
			if not test_all_pths:
				paths = paths[-1:]

			for path in paths:
				subpath = path.split("/")[-1]
				# If I only request testing the last line, don't skip, it is probably retesting
				if not test_all_pths or subpath not in done_checkpoints:
					result = run_single_test(path, config)
					all_results.append((result, subpath))

					with open(output_csv, "a", encoding="UTF-8") as f:
						f.write(f"{subpath},")
						for key in all_metric_names:
							f.write(f"{result[key]},")
						f.write("\n")
						f.flush()

	else:
		print("No checkpoint paths file found or it is empty.")

def run_single_test(checkpoint_path, config):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model_interface = FlowModelInterface(config["module"], device=device, local_rank=None)
	
	if checkpoint_path is not None:
		saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
		state_dict = saved["state_dict"]
		
		# Don't use torch.compile, because the test is fast enough.
		new_state_dict = convert_to_compiled(state_dict=state_dict, local_rank=None, use_compile=False)
		
		model_interface.e2vid_model.load_state_dict(new_state_dict, strict=False)
		print("Loaded checkpoint:", checkpoint_path)

	model_interface.e2vid_model.to(device)

	test_dataloader = create_test_dataloader(config["test_stage"])
	return run_test(model_interface, test_dataloader, device, config)

if __name__ == "__main__":
	main()
