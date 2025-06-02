import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from data.data_interface import make_concat_multi_dataset
from model.train_utils import ModelInterface
from model.train_flow_utils import FlowModelInterface
from utils.util import instantiate_from_config, instantiate_scheduler_from_config
import sys
import yaml
from torch.utils.tensorboard import SummaryWriter
import tqdm
import numpy as np
import datetime
import torch.distributed as dist
torch._dynamo.config.optimize_ddp = False

def convert_to_compiled(state_dict, local_rank, use_compile=False):
	new_dict = {}
	for k, v in state_dict.items():
		parts = k.split(".")
		# First pop out "_orig_mod" and "module"
		if parts[0] == "_orig_mod":
			parts.pop(0)
		if parts[0] == "module":
			parts.pop(0)

		# Then, add the required "_orig_mod" and "module" back
		if local_rank is not None:
			# Should be DDP
			parts.insert(0, "module")
		# Use torch.compile
		if use_compile:
			parts.insert(0, "_orig_mod")

		new_k = ".".join(parts)
		new_dict[new_k] = v

	return new_dict

def setup_process_group(backend="nccl"):
	# Initialize the process group
	dist.init_process_group(
		backend=backend,
		init_method="env://"
	)
	torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_process_group():
	dist.destroy_process_group()

def create_dataloader(dataset, configs, batch_size, local_rank):

	if local_rank is not None:
		# DDP
		sampler = DistributedSampler(dataset)
	else:
		sampler = torch.utils.data.RandomSampler(dataset)

	num_workers = configs.get("num_workers")
	persistent_workers = configs.get("persistent_workers", False)
	pin_memory = configs.get("pin_memory", False)

	dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory, drop_last=True)
	return dataloader

def log_loss(all_loss, writer, batch_idx, epoch, dataloader_len, prefix=""):
	for k, v in all_loss.items():
		writer.add_scalar(f"{prefix}/{k}", v, epoch * dataloader_len + batch_idx)

def train(model_interface, dataloader, optimizer, device, writer, epoch, local_rank, make_preview=False):
	model_interface.e2vid_model.train()
	# Use tqdm to show a progress bar if on GPU=0
	if local_rank is None or local_rank == 0:
		dataloader = tqdm.tqdm(dataloader)
	for batch_idx, batch in enumerate(dataloader):
		optimizer.zero_grad()
		# batch is a dict of tensors
		for k, v in batch.items():
			if isinstance(v, torch.Tensor):
				batch[k] = v.to(device)
		pred = model_interface.forward_sequence(batch, val=False)
		all_loss = model_interface.calc_loss(batch, pred)
		log_loss(all_loss, writer, batch_idx, epoch, len(dataloader), prefix="train")
		loss = all_loss["loss"]
		loss.backward()
		optimizer.step()

		if batch_idx == 0 and make_preview:
			preview = model_interface.make_preview(batch, pred)
			B, T, one, H, W3 = preview.shape
			add_vid_cnt = min(4, B)
			writer.add_video(f"train/preview", preview[:add_vid_cnt], epoch)

def validate(model_interface, dataloader, device, writer, epoch, local_rank):
	model_interface.e2vid_model.eval()
	# Use tqdm to show a progress bar if on GPU=0
	if local_rank is None or local_rank == 0:
		dataloader = tqdm.tqdm(dataloader)
		print("Validation epoch ", epoch)
	
	total_val_loss = []
	for batch_idx, batch in enumerate(dataloader):
		with torch.no_grad():
			for k, v in batch.items():
				if isinstance(v, torch.Tensor):
					batch[k] = v.to(device)
			pred = model_interface.forward_sequence(batch, val=True)
			all_loss = model_interface.calc_loss(batch, pred, remove_flow_loss=True)
			log_loss(all_loss, writer, batch_idx, epoch, len(dataloader),prefix="val")
			total_val_loss.append(all_loss["loss"].item())
			if batch_idx == 0:
				preview = model_interface.make_preview(batch, pred)
				B, T, one, H, W3 = preview.shape
				add_vid_cnt = min(4, B)
				writer.add_video(f"val/preview", preview[:add_vid_cnt], epoch)
				# for b in range(min(4, B)):
				# 	writer.add_video(f"val/preview_{b}", preview[b], epoch)
	return np.mean(total_val_loss)

def train_stage(device, model_interface, configs, cur_epoch, optimizer, lr_scheduler, cur_stage, epochs_of_stages, writer, local_rank):

	stage_config = configs["train_stages"][cur_stage]

	epochs_before = sum(epochs_of_stages[:cur_stage])
	rel_epoch = cur_epoch - epochs_before

	train_dataset = make_concat_multi_dataset(stage_config["dataset"]["train"])
	val_dataset = make_concat_multi_dataset(stage_config["dataset"]["val"])
	train_dataloader = create_dataloader(train_dataset, stage_config["dataset"], stage_config["dataset"]["train_batch_size"], local_rank)
	val_dataloader = create_dataloader(val_dataset, stage_config["dataset"], stage_config["dataset"]["val_batch_size"], local_rank)

	for epoch in range(rel_epoch, stage_config["max_epochs"]):
		print(f"Stage {cur_stage}, Epoch {epochs_before} + {epoch}/{stage_config['max_epochs']}")
		model_interface.set_current_epoch(epoch)

		is_val_epoch = epoch % configs["check_val_every_n_epoch"] == 0 or epoch == stage_config["max_epochs"] - 1

		train(model_interface, train_dataloader, optimizer, device, writer, epoch, local_rank, make_preview=is_val_epoch)
		lr_scheduler.step()
		if is_val_epoch:
			total_loss = validate(model_interface, val_dataloader, device, writer, epoch, local_rank)

			if local_rank is None or local_rank == 0:
				experiment_name = configs["experiment_name"]
				torch.save({
					"state_dict": model_interface.e2vid_model.state_dict(),
					"epoch": epoch + epochs_before,
					"optimizer": optimizer,
					"lr_scheduler": lr_scheduler,
				}, f"checkpoints/{experiment_name}/epoch_{epoch + epochs_before:04d}.pth")
				with open(f"ckpt_paths/{experiment_name}.txt", "a") as f:
					f.write(f"checkpoints/{experiment_name}/epoch_{epoch + epochs_before:04d}.pth\n")
				
				val_loss_txt = os.path.join("tensorboard_logs", experiment_name, "val_loss.txt")
				with open(val_loss_txt, "a") as f:
					# Get current YYYY-MM-DD HH:MM:SS time
					time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
					f.write(f"{time} Epoch {epoch + epochs_before}: {total_loss}\n")


def main(configs):
	# Setup distributed backend and device
	try:
		local_rank = int(os.environ["LOCAL_RANK"])
		setup_process_group(backend="nccl")
		device = torch.device("cuda", local_rank)
	except:
		# Not DDP
		local_rank = None
		device = torch.device("cuda")
	use_compile = configs.get("use_compile", True)

	experiment_name = configs["experiment_name"]
	ckpt_path_file = f"ckpt_paths/{experiment_name}.txt"
	# The path file should be a list of paths to the checkpoints. If it exists & is not empty, load the last checkpoint.
	if os.path.exists(ckpt_path_file) and os.path.getsize(ckpt_path_file) > 0:
		with open(ckpt_path_file, "r") as f:
			ckpt_paths = f.readlines()
			ckpt_paths = [ckpt_path.strip() for ckpt_path in ckpt_paths]
			# Remove all empty lines
			ckpt_paths = [ckpt_path for ckpt_path in ckpt_paths if ckpt_path]
		checkpoint_path = ckpt_paths[-1]
	else:
		# Create ckpt_path_file
		with open(ckpt_path_file, "w") as f:
			pass
		checkpoint_path = None
	
	task = configs.get("task", "e2vid")
	assert task in ["e2vid", "flow"]
	if task == "e2vid":
		model_interface = ModelInterface(configs["module"], device=device, local_rank=local_rank)
	elif task == "flow":
		model_interface = FlowModelInterface(configs["module"], device=device, local_rank=local_rank)

	if local_rank is not None:
		model_interface.e2vid_model = DDP(model_interface.e2vid_model, device_ids=[local_rank])
	if use_compile:
		model_interface.e2vid_model = torch.compile(model_interface.e2vid_model, dynamic=True)

	if checkpoint_path is not None:
		saved = torch.load(checkpoint_path)
		state_dict = convert_to_compiled(saved["state_dict"], local_rank, use_compile)
		model_interface.e2vid_model.load_state_dict(state_dict)
		# Load the epoch number from the .pth file
		cur_epoch = saved["epoch"]
		print("Loaded pretrained from checkpoint: ", checkpoint_path)
		print("Will resume training from epoch ", cur_epoch)
		just_resumed = True
		
	else:
		cur_epoch = 0
		just_resumed = False

	# Decide which stage to start from
	epochs_of_stages = [stage["max_epochs"] for stage in configs["train_stages"]]
	for i, s_epochs in enumerate(epochs_of_stages):
		total_before = sum(epochs_of_stages[:i])
		if cur_epoch < total_before + s_epochs:
			break
	cur_stage = i
	print("Starting from stage ", cur_stage)

	log_dir = os.path.join("tensorboard_logs", configs["experiment_name"])
	os.makedirs(log_dir, exist_ok=True)
	val_loss_txt = os.path.join(log_dir, "val_loss.txt")
	if not os.path.exists(val_loss_txt):
		with open(val_loss_txt, "w") as f:
			pass
	writer = SummaryWriter(log_dir)

	checkpoint_dir = os.path.join("checkpoints", configs["experiment_name"])
	os.makedirs(checkpoint_dir, exist_ok=True)

	for stage_idx in range(cur_stage, len(configs["train_stages"])):
		stage_config = configs["train_stages"][stage_idx]
		optimizer = instantiate_scheduler_from_config(stage_config["optimizer"], model_interface.e2vid_model.parameters())
		lr_scheduler = instantiate_scheduler_from_config(stage_config["lr_scheduler"], optimizer)
		if just_resumed:
			optimizer.load_state_dict(saved["optimizer"].state_dict())
			lr_scheduler.load_state_dict(saved["lr_scheduler"].state_dict())
			# print(optimizer)
			# # Print the current epoch of lr_scheduler
			# print("Current epoch of lr_scheduler: ", lr_scheduler.last_epoch)

		train_stage(device, model_interface, configs, cur_epoch, optimizer, lr_scheduler, stage_idx, epochs_of_stages, writer, local_rank)
		cur_epoch = sum(epochs_of_stages[:stage_idx+1])
		just_resumed = False

	if local_rank is not None:
		cleanup_process_group()

if __name__ == "__main__":
	if len(sys.argv) > 1:
		config_path = sys.argv[1]
	else:
		config_path = "configs/template.yaml"
		
	with open(config_path) as f:
		config = yaml.load(f, Loader=yaml.Loader)

	main(config)