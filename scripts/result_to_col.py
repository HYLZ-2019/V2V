# Tool script that converts a row from a test result to a column (easy to copy & paste into a feishu doc / excel sheet).

import os
import sys

sequences = {
	"HQF": [
		"bike_bay_hdr", 
		"boxes", 
		"desk", 
		"desk_fast", 
		"desk_hand_only", 
		"desk_slow", 
		"engineering_posters", 
		"high_texture_plants", 
		"poster_pillar_1", 
		"poster_pillar_2", 
		"reflective_materials", 
		"slow_and_fast_desk", 
		"slow_hand", 
		"still_life"
	],
	"EVAID": [
		"ball", 
		"bear",
		"box", 
		"building", 
		"outdoor", 
		"playball", 
		"room1",
		"sculpture", 
		"toy", 
		"traffic",
		"wall"
	],
	"IJRR": [
		"boxes_6dof",
		"calibration",
		"dynamic_6dof",
		"office_zigzag",
		"poster_6dof",
		"shapes_6dof",
		"slider_depth"
	],
	"MVSEC": [
		"indoor_flying1",
		"indoor_flying2",
		"indoor_flying3",
		"indoor_flying4",
		"outdoor_day1",
		"outdoor_day2"
	]
}

all_metric_names = []
for dataset in ["HQF", "EVAID"]:
	for seqname in sequences[dataset]:
		for metric in ["MSE", "SSIM", "LPIPS"]:
			all_metric_names.append(f"{dataset}/{seqname}/{metric}")

with open("debug/col_heads.txt", "w") as f:
	for key in all_metric_names:
		f.write(f"{key}\n")

# line_n is the line number in vscode
def extract_line(file_name, line_n):
	with open(file_name, "r", encoding="UTF-8") as f:
		lines = f.readlines()
		head = lines[0].split(",")
		
		data = {}
		dataline = lines[line_n-1].split(",")
		for i in range(len(dataline)):
			data[head[i]] = dataline[i]
		
		of_path = os.path.join(os.path.dirname(file_name), f"col_from_line_{line_n:03d}_{dataline[0]}.txt")

		with open(of_path, "w", encoding="UTF-8") as of:
			for key in all_metric_names:
				val = float(data[key])
				if "SSIM" in key:
					val = -val
				of.write(f"{val:.03f}\n")

		# In avg_metrics_from_line_{line_n}.txt, each line is average over all sequence, e.g. HQF/MSE.
		metric_pth = os.path.join(os.path.dirname(file_name), f"avg_metrics_from_line_{line_n:03d}_{dataline[0]}.txt")
		with open(metric_pth, "w") as of:
			for dataset in ["HQF", "EVAID", "IJRR", "MVSEC"]:
				for metric in ["MSE", "SSIM", "LPIPS"]:
					vals = [v for k, v in data.items() if dataset in k and metric in k]
					avg = sum([float(v) for v in vals]) / len(vals)
					of.write(f"{avg:.03f}\n")

if __name__ == "__main__":
	file_name = sys.argv[1]
	line_n = int(sys.argv[2])
	extract_line(file_name, line_n)