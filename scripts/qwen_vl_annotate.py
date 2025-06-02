# The annotation results can be found along with the V2V checkpoints (checkpoints/webvid_annots.txt).

import cv2
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import glob
import tqdm
import os
import json
import concurrent.futures
import threading

questions = [
	("Is this image a photo without any post-production artefacts (such as subtitles/CG/...)? (True/False) ", "real"),
	("Is there an outdoor scene in this image? (True/False)", "outdoor"),
	("Is there an indoor scene in this image? (True/False)", "indoor"),
	("Is it daytime in this image? (True/False)", "day"),
	("Is it nighttime in this image? (True/False)", "night"),
	("Is there water in this image? (True/False)", "water"),
	("Are there humans in this image? (True/False)", "human"),
	("Is the sky in this image? (True/False)", "sky"),
	("Is this image blank (pure white / black / ... without any significant object)? (True/False)", "blank"),
	("Is there out-of-focus blur in this image? (True/False)", "defocus"),
	("Is there motion blur in this image? (True/False)", "motion"),
	("Is there any object with text (such as a book cover) in this image? (True/False)", "text"),
	("Describe the content of the photo.", "description")
]

# 1. 加载本地模型和tokenizer
model_dir = "pretrained/qwen_2_5_vl"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
	model_dir, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
)
processor = AutoProcessor.from_pretrained(model_dir, use_fast=True, padding_side="left")


def infer(img_path, text):
	messages = [
		{
			"role": "user",
			"content": [
				{
					"type": "image",
					"image": img_path,
				},
				{"type": "text", "text": text},
			],
		}
	]

	# Preparation for inference
	text = processor.apply_chat_template(
		messages, tokenize=False, add_generation_prompt=True
	)
	image_inputs, video_inputs = process_vision_info(messages)
	inputs = processor(
		text=[text],
		images=image_inputs,
		videos=video_inputs,
		padding=True,
		return_tensors="pt",
	)
	inputs = inputs.to("cuda")

	# Inference: Generation of the output
	generated_ids = model.generate(**inputs, max_new_tokens=128)
	generated_ids_trimmed = [
		out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
	]
	output_text = processor.batch_decode(
		generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
	)
 
	return output_text

def infer_batch(img_path):
	messages = [
			[{
				"role": "user",
				"content": [
					{
						"type": "image",
						"image": img_path,
					},
					{"type": "text", "text": question},
				],
			}
		] for question, key in questions
    ]

	# Preparation for batch inference
	texts = [
		processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
		for msg in messages
	]
	image_inputs, video_inputs = process_vision_info(messages)
	inputs = processor(
		text=texts,
		images=image_inputs,
		videos=video_inputs,
		padding=True,
		return_tensors="pt",
	)
	inputs = inputs.to("cuda")

	# Batch Inference
	generated_ids = model.generate(**inputs, max_new_tokens=128)
	generated_ids_trimmed = [
		out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
	]
	output_texts = processor.batch_decode(
		generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
	)
	print(output_texts)
 
	return output_texts


def annotate_img(image_path):
	img_id = os.path.basename(image_path).split(".")[0]
	annot = {"id": img_id}

	all_answers = infer_batch(image_path)
	for idx, (question, key) in enumerate(questions):
		#response = infer(image_path, question)[0]
		response = all_answers[idx]
		print(response)
		if key == "description":
			annot[key] = response
		else:
			if "true" in response.lower():
				annot[key] = True
			else:
				annot[key] = False
	return annot

def process_image(image_path, lock, annot_path):
	try:
		text = annotate_img(image_path)
		with lock:
			with open(annot_path, "a", encoding="UTF-8") as f:
				f.write(str(text) + "\n")
				f.flush()  # 立即写入磁盘
	except Exception as e:
		print("Exception", e, "with", image_path, ":", e)

def main():
	all_img_paths = sorted(glob.glob("../data/webvid_imgs/*.png"))
	annot_path = "../data/webvid/annots.txt"

	# # 清空文件
	# with open(annot_path, "w") as f:
	# 	pass

	lock = threading.Lock()

	with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
		futures = [executor.submit(process_image, ip, lock, annot_path) for ip in all_img_paths]
		for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(all_img_paths)):
			future.result()  # 获取结果，如果出现异常，会在这里抛出

if __name__ == "__main__":
	main()