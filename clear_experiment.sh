# Usage: ./clear_experiment.sh <experiment_name>
# rm -r tensorboard_logs/{experiment_name}
# rm ckpt_paths/{experiment_name}.txt
# rm -r checkpoints/{experiment_name}

#!/bin/bash

# Check if experiment name is provided
if [ -z "$1" ]; then
	echo "Usage: $0 <experiment_name>"
	exit 1
fi

experiment_name=$1

# Define paths
tensorboard_dir="tensorboard_logs/${experiment_name}"
ckpt_file="ckpt_paths/${experiment_name}.txt"

# Remove tensorboard directory if it exists
if [ -d "$tensorboard_dir" ]; then
	echo "Removing directory: $tensorboard_dir"
	rm -r "$tensorboard_dir"
else
	echo "Directory not found: $tensorboard_dir"
fi

# Remove checkpoint file if it exists
if [ -f "$ckpt_file" ]; then
	echo "Removing file: $ckpt_file"
	rm "$ckpt_file"
else
	echo "File not found: $ckpt_file"
fi

# Remove checkpoints directory if it exists
checkpoints_dir="checkpoints/${experiment_name}"
if [ -d "$checkpoints_dir" ]; then
	echo "Removing directory: $checkpoints_dir"
	rm -r "$checkpoints_dir"
else
	echo "Directory not found: $checkpoints_dir"
fi

echo "Cleanup for experiment '$experiment_name' complete."