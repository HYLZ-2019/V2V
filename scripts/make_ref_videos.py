import cv2
import numpy as np
import glob
import os
import ffmpeg
import sys

def make_video(src_dirs, dataset, seqname, dst_dir):
	# Read all images in {src_dir}/{dataset}/{seqname}/*.png
	src_img_paths = [
		sorted(glob.glob(f"{src_dir}/{dataset}/{seqname}/*.png"))
		for src_dir in src_dirs
	]
	assert all(len(src_img_paths[0]) == len(src_img_paths[i]) for i in range(1, len(src_dirs)))
	T = len(src_img_paths[0])

	# Get image dimensions from the first image
	img0 = cv2.imread(src_img_paths[0][0])
	H, W, _ = img0.shape
	pad_W = W % 4
	pad_H = H % 4
	H += pad_H
	W += pad_W
	
	os.makedirs(f"{dst_dir}/{dataset}", exist_ok=True)
	output_path = f"{dst_dir}/{dataset}/{seqname}.mp4"
	
	# Start ffmpeg process with highest quality settings
	process = (
		ffmpeg
		.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{W*len(src_dirs)}x{H}', r=24)
		.output(output_path,
			vcodec='libx264',       # Specify H.264 codec
			pix_fmt='yuv420p',      # Keep yuv420p for compatibility
			r=24,
			movflags='faststart',   # Allow streaming
			tune='film',            # Optimize for high quality video
			**{'b:v': '0'}          # Let CRF control quality
		)
		.overwrite_output()
		.run_async(pipe_stdin=True)
	)
	
	try:
		# Process and write frames one by one
		for t in range(T):
			imgs = [cv2.imread(src_img_paths[i][t]) for i in range(len(src_dirs))]
			imgs = [
				cv2.copyMakeBorder(img, 0, pad_H, 0, pad_W, cv2.BORDER_CONSTANT, value=[0, 0, 0])
				for img in imgs
			]
			img = np.concatenate(imgs, axis=1)
			# Convert from BGR to RGB for ffmpeg
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			process.stdin.write(img.tobytes())
	finally:
		# Ensure pipe is closed and process ends
		process.stdin.close()
		process.wait()

def make_videos(experiment_name):
	ref_dir = "results/e2vid++_original"
	my_dir = f"results/{experiment_name}"
	dst_dir = f"videos/{experiment_name}"
	gt_dir = "results/gt_images"

	datasets = ["EVAID", "IJRR", "HQF", "MVSEC"]
	for dataset in datasets:
		seqnames = os.listdir(f"{ref_dir}/{dataset}")
		for seqname in seqnames:
			make_video([my_dir, ref_dir, gt_dir], dataset, seqname, dst_dir)

def make_videos_comb(out_name, experiment_name_list):

	all_dirs = [f"results/{experiment_name}" for experiment_name in experiment_name_list]

	datasets = ["EVAID", "IJRR", "HQF", "MVSEC"]
	for dataset in datasets:
		seqnames = os.listdir(f"{all_dirs[0]}/{dataset}")
		for seqname in seqnames:
			make_video(all_dirs, dataset, seqname, f"videos/{out_name}")

if __name__ == "__main__":
	# Take the experiment name as argument
	if len(sys.argv) > 1:
		experiment_name = sys.argv[1]
		make_videos(experiment_name)
	else:
		print("Please provide the experiment name as argument.")