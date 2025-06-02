# Used to generate random thresholds for fixed-threshold ablation experiments.

import numpy as np

def ran_thres(threshold_range=[0.05, 2], max_thres_pos_neg_gap=1.5):
	thres_1 = np.random.uniform(*threshold_range)
	pos_neg_gap = np.random.uniform(1, max_thres_pos_neg_gap)
	thres_2 = thres_1 * pos_neg_gap
	if np.random.rand() > 0.5:
		pos_thres = thres_1
		neg_thres = thres_2
	else:
		pos_thres = thres_2
		neg_thres = thres_1
	return pos_thres, neg_thres

def process_file(input_file):
	# Original content of file:
	# 000401_000450/5876366.mp4 225
	# 000351_000400/1050217609.mp4 885

	# Target content of file:
	# 000401_000450/5876366.mp4 225 0.07 0.075
	# 000351_000400/1050217609.mp4 885 0.05 0.06
	with open(input_file, 'r') as f:
		lines = f.readlines()
	processed_lines = []

	for line in lines:
		line = line.strip()
		if not line:
			continue
		parts = line.split()
		video_path = parts[0]
		frame_num = parts[1]
		pos_thres, neg_thres = ran_thres()
		processed_line = f"{video_path} {frame_num} {pos_thres:.3f} {neg_thres:.3f}"
		processed_lines.append(processed_line)

	# Write the processed lines to a new file
	with open(input_file, 'w') as f:
		for processed_line in processed_lines:
			f.write(processed_line + '\n')

if __name__ == "__main__":
	process_file("config/webvid10000_unfiltered.txt")