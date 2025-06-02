import os
import sys
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import glob
import time

size_guidance = {
    event_accumulator.COMPRESSED_HISTOGRAMS: 0,  # 跳过压缩直方图
    event_accumulator.IMAGES: 0,                 # 跳过图像数据
    event_accumulator.AUDIO: 0,                  # 跳过音频数据
    event_accumulator.SCALARS: 10000,            # 仅保留标量数据
    event_accumulator.HISTOGRAMS: 0,             # 跳过普通直方图
    event_accumulator.TENSORS: 0,                # 跳过张量数据
    event_accumulator.GRAPH: 0,                  # 跳过计算图数据
    event_accumulator.META_GRAPH: 0              # 跳过元图数据
}

def process(experiment_name, epochs_per_val):
	log_list = sorted(glob.glob(f"tensorboard_logs/{experiment_name}/events.out.tfevents*"))

	tags_to_track = ["val/perceptual_loss/evaid", "val/perceptual_loss/hqf", "val/perceptual_loss/ijrr", "val/perceptual_loss/mvsec"]

	all_events = {tag: [] for tag in tags_to_track}

	for in_pth in log_list[:]:
		try:
			ea = event_accumulator.EventAccumulator(in_pth, size_guidance=size_guidance)
			ea.Reload()
			for tag in tags_to_track:
				events = ea.scalars.Items(tag)
				all_events[tag].extend(events)
		except:
			pass

	avg_metric_per_epoch = {}

	for tag in tags_to_track:
		# Each metric has different number of steps due to mixed dataset
		steps = [event.step for event in all_events[tag]]
		print(steps)
		steps = np.array(steps)
		epochs = np.zeros_like(steps)
		ep = 0
		for i in range(0, len(steps)):
			# if steps[i] - steps[i-1] > 500:
			# 	ep += 1
			# epochs[i] = ep
			epochs[i] = steps[i] // (381*epochs_per_val)
		max_ep = max(epochs) + 1
		metric = np.array([event.value for event in all_events[tag]])
		avg_metric_per_epoch[tag] = np.zeros((max_ep))

		for i in range(max_ep):
			sub_metrics = metric[epochs == i]
			if len(sub_metrics) > 0:
				avg_metric_per_epoch[tag][i] = np.mean(sub_metrics)
			else:
				avg_metric_per_epoch[tag][i] = 1e6

	num_epochs = len(avg_metric_per_epoch[tags_to_track[0]])
	for tag in tags_to_track:
		assert len(avg_metric_per_epoch[tag]) == num_epochs, f"Length mismatch for {tag}: {len(avg_metric_per_epoch[tag])} vs {num_epochs}"
	
	loss_output_file = os.path.join("tensorboard_logs", experiment_name, "calc_val_loss_per_checkpoint.txt")
	all_total_loss = []
	with open(loss_output_file, "w") as f:
		for i in range(num_epochs):
			# IMPORTANT: Only use the two
			total_loss = avg_metric_per_epoch["val/perceptual_loss/evaid"][i] + avg_metric_per_epoch["val/perceptual_loss/hqf"][i]
			#total_loss = avg_metric_per_epoch["val/perceptual_loss/hqf"][i] + avg_metric_per_epoch["val/perceptual_loss/ijrr"][i] + avg_metric_per_epoch["val/perceptual_loss/mvsec"][i]
			all_total_loss.append(total_loss)
			f.write(f"{total_loss:.03f}\n")
	
	best_idx = np.argmin(all_total_loss)
	print("best_idx: ", best_idx)
	all_checkpoints = sorted(glob.glob(f"checkpoints/{experiment_name}/*.pth"))
	print("Selected checkpoint: ", all_checkpoints[best_idx])

if __name__ == "__main__":
	# Take the experiment name as argument
	if len(sys.argv) > 2:
		epochs_per_val = int(sys.argv[2])
	else:
		epochs_per_val = 1

	if len(sys.argv) > 1:
		experiment_name = sys.argv[1]
		start = time.time()
		process(experiment_name, epochs_per_val)
		end = time.time()
		print("Used seconds: ", end-start)
	else:
		print("Please provide the experiment name as argument.")