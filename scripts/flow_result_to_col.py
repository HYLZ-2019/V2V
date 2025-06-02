# Tool script that converts a row from a test result to a column (easy to copy & paste into a feishu doc / excel sheet).

import os
import sys

sequences = {
	"MVSEC": [
		"indoor_flying1",
		"indoor_flying2",
		"indoor_flying3",
		"outdoor_day1",
		"outdoor_day2"
	]
}

all_metric_names = []
for dataset in ["MVSEC"]:
	for seqname in sequences[dataset]:
		for metric in ["dense_EPE", "dense_3PE", "sparse_EPE", "sparse_3PE"]:
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
				of.write(f"{val:.03f}\n")

if __name__ == "__main__":
	file_name = sys.argv[1]
	line_n = int(sys.argv[2])
	extract_line(file_name, line_n)