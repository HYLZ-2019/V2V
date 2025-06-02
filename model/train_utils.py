import torch
import torchmetrics
import torch.nn.functional as F

import traceback
import collections
import numpy as np
from skimage.metrics import structural_similarity as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from collections import defaultdict


from utils.util import instantiate_from_config
from model.loss import l2_loss, perceptual_loss, l1_loss, ssim_loss, temporal_consistency_loss
# Names of each data source: ('esim', 'ijrr', ...)
from utils.data import data_sources
from PerceptualSimilarity.models import PerceptualLoss
from torchvision.models.optical_flow import raft_small, raft_large
import matplotlib.pyplot as plt

def load_raft(raft_name, local=False, device=None):
	if local:
		# Save .pth on machine with network, then load it on machine without network
		if raft_name == "raft_small":
			raft_model = raft_small(pretrained=False)
			state_dict = torch.load(
				"pretrained/raft_small.pth", 
				map_location="cpu",  # Load to CPU first
				weights_only=True
			)
			raft_model.load_state_dict(state_dict)
		elif raft_name == "raft_large":
			raft_model = raft_large(pretrained=False)
			state_dict = torch.load(
				"pretrained/raft_large.pth", 
				map_location="cpu",  # Load to CPU first
				weights_only=True
			)
			raft_model.load_state_dict(state_dict)
	else:
		if raft_name == "raft_small":
			raft_model = raft_small(pretrained=True)
		elif raft_name == "raft_large":
			raft_model = raft_large(pretrained=True)
		else:
			raise ValueError(f"Unknown RAFT model name: {raft_name}")
		
	raft_model.eval()  # Set to evaluation mode

	if device is not None:
		raft_model = raft_model.to(device)

	return raft_model

def inference_raft(raft_model, raft_num_flow_updates, img1, img2):
	# Input range is [0, 1], RAFT accepts [-1, 1]
	img1 = img1 * 2 - 1
	img2 = img2 * 2 - 1
	# Pad the image to be divisible by 8
	PAD_SIZE = 8
	B, T, C, H, W = img1.shape
	PAD_H = int(np.ceil(H / PAD_SIZE) * PAD_SIZE)
	PAD_W = int(np.ceil(W / PAD_SIZE) * PAD_SIZE)
	PAD_H = max(PAD_H, 128)
	PAD_W = max(PAD_W, 128)
	padded_img1 = torch.zeros((B, T, C, PAD_H, PAD_W), device=img1.device)
	padded_img2 = torch.zeros((B, T, C, PAD_H, PAD_W), device=img2.device)
	padded_img1[:, :, :, :H, :W] = img1
	padded_img2[:, :, :, :H, :W] = img2
	img1_flat = torch.flatten(padded_img1, start_dim=0, end_dim=1)  # [(B*T), C, H, W]
	img2_flat = torch.flatten(padded_img2, start_dim=0, end_dim=1)  # [(B*T), C, H, W]
	if C == 1:
		img1_flat = img1_flat.repeat(1, 3, 1, 1)
		img2_flat = img2_flat.repeat(1, 3, 1, 1)
	# Use RAFT to calculate flow
	flow_flat = raft_model(img1_flat, img2_flat, num_flow_updates=raft_num_flow_updates)[-1]
	flow = torch.reshape(flow_flat, (B, T, 2, PAD_H, PAD_W)).detach()  # [B, T, 2, H, W]
	flow = flow[:, :, :, :H, :W]  # Remove padding
	return flow

def norm(arr):
	return arr / 255.0 * 2 - 1

def printshapes(batch):
	for k, v in batch.items():
		if isinstance(v, torch.Tensor):
			print(k, v.shape)
			
# Use to check which step of the model is causing NaN
def nan_hook(self, inp, output):

	all_outputs = []
	# Recursively take out all nested tensors
	def flatten_nested_tensors(x):
		if isinstance(x, torch.Tensor):
			all_outputs.append(x)
		elif isinstance(x, tuple) or isinstance(x, list):
			for y in x:
				flatten_nested_tensors(y)
		elif isinstance(x, dict):
			for k, v in x.items():
				flatten_nested_tensors(v)
		else:
			raise RuntimeError("Unknown type", type(x))
	
	flatten_nested_tensors(output)

	for i, out in enumerate(all_outputs):
		nan_mask = torch.isnan(out)
		if nan_mask.any():
			print("In", self.__class__.__name__)
			traceback.print_stack()
			raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

def normalize(x):
	maxval = x.quantile(0.99)
	minval = x.quantile(0.01)
	div = max(maxval - minval, 1e-5)
	x = torch.clamp(x, minval, maxval)
	return (x - minval) / div

def viz_voxel(voxel):
	v = voxel.sum(axis=0)
	v = (v + 1) / 2 * 255
	v = torch.clamp(v, 0, 255)
	return v#.astype(torch.uint8)

def normalize_nobias(x):
	n = int(x.numel() * 0.99)
	pos_maxval = max(x.view(-1).kthvalue(n).values.item(), 1e-3)
	neg_maxval = max(((-x).view(-1).kthvalue(n)).values.item(), 1e-3)
	x = torch.clamp(x, -neg_maxval, pos_maxval)
	# Let positive part /= pos_maxval, negative part /= neg_maxval
	x = torch.where(x > 0, x / pos_maxval, x / neg_maxval)
	return x / 2 + 0.5

def concat_imgs(imgs):
	imgs_3c = []
	for img in imgs:
		img = img.squeeze()
		if len(img.shape) == 2:
			img = torch.stack([img, img, img], axis=0)
		imgs_3c.append(img)
	img_cat = torch.cat(imgs_3c, axis=2)
	return img_cat

def normalize_batch_voxel(voxel):
	# Assert that the shape is (B, T, C, H, W). Max and min are [B, 1, 1, 1, 1].
	assert len(voxel.shape) == 5
	B, T, C, H, W = voxel.shape
	# torch.quantile can only operate on one dim. Reshape it first
	voxel_flat = voxel.reshape((B, -1))
	max_k = int(0.99 * voxel_flat.shape[1])
	min_k = int(0.01 * voxel_flat.shape[1])
	pos_max = torch.kthvalue(voxel_flat, max_k, dim=1).values
	neg_max = -torch.kthvalue(voxel_flat, min_k, dim=1).values
	#pos_max = torch.quantile(voxel_flat, 0.99, dim=1)
	#neg_max = -torch.quantile(voxel_flat, 0.01, dim=1)
	pos_max = pos_max.reshape((B, 1, 1, 1, 1))
	neg_max = neg_max.reshape((B, 1, 1, 1, 1))
	# Ensure pos_max and neg_max are at least 1
	pos_max = torch.clamp(pos_max, min=1)
	neg_max = torch.clamp(neg_max, min=1)   
	# Normalize the voxel
	norm_voxel = torch.where(voxel > 0, voxel / pos_max, voxel / neg_max)
	return norm_voxel


class ModelInterface():
	def __init__(self, configs, device, local_rank):
		
		self.configs = configs
		self.device = device
		self.local_rank = local_rank

		self.e2vid_model = instantiate_from_config(configs["model"]).to(device)

		for submodule in self.e2vid_model.modules():
			submodule.register_forward_hook(nan_hook)

		# RAFT is used for generation of optical flow when real optical flow is not available and we also need to calculate the temporal consistency loss.
		# Else, don't bother to load it.
		if configs["loss"].get("temporal_consistency_weight", 0) > 0:
			self.optical_flow_source = configs["loss"].get("optical_flow_source", "gt")
			assert self.optical_flow_source in ["raft_small", "raft_large", "gt", "zeros"], f"Unknown optical flow source: {self.optical_flow_source}"
	
			if self.optical_flow_source == "raft_small":
				self.raft_model = load_raft("raft_small", device=device)
			elif self.optical_flow_source == "raft_large":
				self.raft_model = load_raft("raft_large", device=device)

			self.raft_num_flow_updates = configs["loss"].get("raft_num_flow_updates", 12)

		self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=255)
		self.loss_functions = self.load_loss_functions(configs["loss"])

		# Not the most modern LPIPS model; used to keep metric consistency with previous papers
		self.test_lpips_fn = PerceptualLoss(net="alex")

		self.normalize_voxels = configs.get("normalize_voxels", False)

		self.current_epoch = 0
		self.hyper_epochs = configs.get("hyper_epochs", 0)

		self.pred_channels = configs.get("pred_channels", 1)

		self.is_nernet = configs.get("is_nernet", False)

	def set_current_epoch(self, epoch):
		self.current_epoch = epoch
		
	def compute_metrics(self, pred, batch):
		# Calculate metrics for test.
		sequence_name = batch["sequence_name"][0]
		data_source_idx = batch["data_source_idx"][0]
		data_source = data_sources[data_source_idx] 
		sequence_name = batch["sequence_name"][0][0]
		log_prefix = f"{data_source.upper()}/{sequence_name}"

		B, T, C, H, W = batch["frame"].shape
		assert B == 1, "Batch size must be 1 for testing."
		
		metrics = defaultdict(list)

		for t in range(T):
			# Calculate MSE, SSIM and LPIPS.
			# Use exactly the same calculating method as in ET-Net code.
			# ET-Net uses skimage.metrics.structural_similarity for SSIM, which has default window size 7. The torchvision version has default window size 11.
			# The metrics are calculated in the [0, 1] range.
			pred_image = pred[0, t, :, :, :]
			#pred_image = torch.clamp(pred_image, 0, 255)
			pred_image = pred_image / 255
			image = batch["frame"][0, t, :, :, :] / 255

			mse = F.mse_loss(pred_image, image).item()
			lpips = self.test_lpips_fn(pred_image, image, normalize=True).item()

			pred_image = pred_image.detach().cpu().numpy().squeeze()
			image = image.detach().cpu().numpy().squeeze()

			# According to the notes in https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity, we should set data_range=1 to SSIM because the input values are in [0, 1], while the data_range will be defaulted to 2 ([-1, 1]) if not set. However, the metrics in the previous papers (e.g. ET-Net) were calculated with the wrong data_range, so we'll follow them to keep consistency.
			ssim = SSIM(pred_image, image, data_range=2)
			
			metrics[f"{log_prefix}/MSE"].append(mse)
			metrics[f"{log_prefix}/LPIPS"].append(lpips)
			metrics[f"{log_prefix}/SSIM"].append(ssim)

		return metrics

	def load_loss_functions(self, cfgs):
		funcs = []
		if cfgs.get("lpips_weight", 0) != 0:
			lpips_weight = cfgs["lpips_weight"]
			self.perceptual_loss = perceptual_loss(weight=lpips_weight, net=cfgs["lpips_type"])
			funcs.append(
				self.perceptual_loss
			)
		if cfgs.get("l2_weight", 0) != 0:
			l2_weight = cfgs["l2_weight"]
			funcs.append(
				l2_loss(weight=l2_weight)
			)
		if cfgs.get("l1_weight", 0) != 0:
			l1_weight = cfgs["l1_weight"]
			funcs.append(
				l1_loss(weight=l1_weight)
			)
		if cfgs.get("ssim_weight", 0) != 0:
			ssim_weight = cfgs["ssim_weight"]
			funcs.append(
				ssim_loss(weight=ssim_weight, model=self.ssim)
			)
		if cfgs.get("temporal_consistency_weight", 0) != 0:
			temporal_consistency_weight = cfgs["temporal_consistency_weight"]
			temporal_consistency_L0 = cfgs.get("temporal_consistency_L0", 1)
			funcs.append(
				temporal_consistency_loss(weight=temporal_consistency_weight, L0=temporal_consistency_L0)
			)
		return funcs
	
	def forward_sequence(self, sequence, reset_states=True, test=False, val=False):
		# If there is no ground truth optical flow, predict some.
		# Testing does not require flow
		# Validation should have flow, but calculation of flow makes GPU OOM (because 720p EVAID x 80-img sequence is big)
		if self.configs["loss"].get("temporal_consistency_weight", 0) > 0 and not test and not val:
			if self.optical_flow_source == "gt":
				assert "flow" in sequence
			elif self.optical_flow_source in ["raft_small", "raft_large"]:
				# Don't calculate flow for the frames before L0, because they won't be used in the loss.
				tcloss_L0 = self.configs["loss"].get("temporal_consistency_L0", 1)		
				img1 = sequence["frame"][:, tcloss_L0-1:-1, :, :, :]
				img2 = sequence["frame"][:, tcloss_L0:, :, :, :]
				B, Tm1, C, H, W = img1.shape
				flow = inference_raft(
					self.raft_model, 
					self.raft_num_flow_updates, 
					img1, img2
				)
				# Concat zero frame to the beginning
				flow = torch.cat([torch.zeros(B, tcloss_L0, 2, H, W, device=sequence["frame"].device), flow], dim=1)
				sequence["flow"] = flow
			elif self.optical_flow_source == "zeros":
				# Create a zero flow tensor
				B, T, C, H, W = sequence["frame"].shape
				sequence["flow"] = torch.zeros((B, T, 2, H, W), device=sequence["frame"].device)

		hyper_gt = self.current_epoch < self.hyper_epochs and not val # Only in training, mix the ground truth with the previous frame for HyperE2VID.

		if reset_states:
			if self.local_rank is not None:
				self.e2vid_model.module.reset_states()
			else:
				self.e2vid_model.reset_states()

		if self.is_nernet:
			return self.forward_sequence_nernet(sequence)

		B, T, C, H, W = sequence["events"].shape
		if self.normalize_voxels:
			sequence["events"] = normalize_batch_voxel(sequence["events"])

		PAD = 16
		padded_h = int(np.ceil(H / PAD) * PAD)
		padded_w = int(np.ceil(W / PAD) * PAD)
		padded_events = torch.zeros((B, T, C, padded_h, padded_w), device=sequence["events"].device)
		padded_events[:,:,:,:H,:W] = sequence["events"]

		if hyper_gt:
			hyper_beta = 1 - (self.current_epoch / self.hyper_epochs)
			# if np.random.rand() > self.current_epoch / self.hyper_epochs:
			# 	hyper_beta = 1
			# else:
			# 	hyper_beta = 0
			padded_gt = torch.zeros((B, T, 1, padded_h, padded_w), device=sequence["events"].device)
			padded_gt[:,:,:,:H,:W] = sequence["frame"]

		pred_imgs = torch.zeros((B, T, self.pred_channels, H, W), device=sequence["events"].device)
		
		for t in range(T):
			if hyper_gt:
				# For HyperE2VID
				pred = self.e2vid_model(event_tensor=padded_events[:, t, :, :, :], gt_image=padded_gt[:, t, :, :, :], beta=hyper_beta)
			else:
				pred = self.e2vid_model(padded_events[:, t, :, :, :])
			pred_imgs[:, t, :, :, :] = pred['image'][:, :, :H, :W]
		
		# pred_imgs should be in [0, 1]
		return pred_imgs
	
	def forward_sequence_nernet(self, sequence):
		# sequence["events"] is a list with T (1, N, 5) raw event streams
		B, T, C_, H, W = sequence["frame"].shape
		assert T == len(sequence["events"])
		assert B == 1 # Only support batch size 1 for NER-Net

		self.e2vid_model.set_resolution(H, W)

		pred_imgs = torch.zeros((1, T, self.pred_channels, H, W), device=sequence["frame"].device)
	
		for t in range(T):
			evs = sequence["events"][t][0].to(sequence["frame"].device)  # [N, 5]
			pred = self.e2vid_model(evs)
			pred_img = pred[0]['image']
			pred_imgs[:, t, :, :, :] = pred_img[0, :, :H, :W]

		plt.imshow(pred_img[0, 0, :H, :W].detach().cpu().numpy())
		plt.colorbar()
		plt.savefig("pred.png")
		plt.close()

		plt.imshow(pred[1][0].sum(axis=0).detach().cpu().numpy())
		plt.colorbar()
		plt.savefig("pred_voxel.png")
		plt.close()

		
		# pred_imgs should be in [0, 1]
		return pred_imgs
	
	def calc_loss(self, sequence, pred, remove_flow_loss=False):
		# pred should be in [0, 1]
		B, T, C, H, W = sequence["events"].shape
		losses = {}

		loss_functions_list = []
		for loss_ftn in self.loss_functions:
			if not remove_flow_loss or not "consistency" in loss_ftn.__class__.__name__:
				loss_functions_list.append(loss_ftn)

		for loss_ftn in loss_functions_list:
			loss_name = loss_ftn.__class__.__name__
			losses[loss_name] = torch.zeros((B, T), device=self.device)

		final_losses = collections.defaultdict(lambda: 0)
		max_val = pred.max()
		min_val = pred.min()
		pred_var = torch.var(pred)
		final_losses["pred_max_val"] = max_val.item()
		final_losses["pred_min_val"] = min_val.item()
		final_losses["pred_var"] = pred_var.item()
	
		for t in range(T):
			# Calculate the losses. Loss is [b], because sequences in one batch may be from different data sources.
			image = sequence["frame"][:, t, :, :, :]
			#print("Max and min of image: ", image.max().item(), image.min().item())
			pred_img = pred[:, t, :, :, :]
			#wprint("Max and min of pred_img: ", pred_img.max().item(), pred_img.min().item())

			for loss_ftn in loss_functions_list:
				#start = time.time()
				loss_name = loss_ftn.__class__.__name__
				if isinstance(loss_ftn, perceptual_loss):
					ls = loss_ftn(pred_img, image, reduce_batch=False)
				elif isinstance(loss_ftn, l2_loss):
					ls = loss_ftn(pred_img, image, reduce_batch=False)
				elif isinstance(loss_ftn, l1_loss):
					ls = loss_ftn(pred_img, image, reduce_batch=False)
				elif isinstance(loss_ftn, ssim_loss):
					raise NotImplementedError
				elif isinstance(loss_ftn, temporal_consistency_loss):
					# image is [B, 1, H, W]. temporal_consistency_loss wants [B, C, H, W].
					ls = loss_ftn(t, image, pred_img, sequence["flow"][:, t], output_images=False, reduce_batch=False)
				
				losses[loss_name][:, t] = ls
				#time_usage[loss_name] += time.time() - start
		
		data_source_indices = sequence['data_source_idx']  # 形状为 (B,)
		unique_sources = torch.unique(data_source_indices)
		
		for loss_ftn in loss_functions_list:
			loss_name = loss_ftn.__class__.__name__
			loss_per_b = torch.mean(losses[loss_name], dim=1)  # Average over T
			
			for idx in unique_sources:
				idx = idx.item()
				data_source = data_sources[idx]
				mask = (data_source_indices == idx)
				total_loss = loss_per_b[mask].sum()
				
				final_losses[f"{loss_name}/{data_source}"] += total_loss.item()
				final_losses[f"loss/{data_source}"] += total_loss.item()
				final_losses[f"{loss_name}"] += total_loss.item()
				final_losses[f"loss"] += total_loss

		# Calculate the average loss
		for idx in unique_sources:
			data_source = data_sources[idx.item()]
			count = (data_source_indices == idx).sum().item()
			for loss_ftn in loss_functions_list:
				loss_name = loss_ftn.__class__.__name__
				final_losses[f"{loss_name}/{data_source}"] /= count
			final_losses[f"loss/{data_source}"] /= count
		final_losses["loss"] /= B

		for loss_ftn in loss_functions_list:
			final_losses[f"{loss_name}"] /= B

		return final_losses

	def make_preview(self, batch, pred):
		"""
		Get visualizations of the predictions and the ground truth.
		"""
		B, T, C, H, W = pred.shape
		event_vis = normalize_nobias(torch.sum(batch["events"], dim=2, keepdim=True))*255 # B, T, 1, H, W
		pred_vis = pred*255 # B, T, C, H, W
		gt_vis = batch["frame"]*255 # B, T, 1, H, W

		# Convert to 3 channels
		event_vis = event_vis.repeat(1, 1, 3, 1, 1)
		if gt_vis.shape[2] == 1:
			gt_vis = gt_vis.repeat(1, 1, 3, 1, 1)
		gt_vis = gt_vis.flip(dims=[2])  # Show RGB, not BGR
		if pred_vis.shape[2] == 1:
			pred_vis = pred_vis.repeat(1, 1, 3, 1, 1)
		pred_vis = pred_vis.flip(dims=[2])
		vis = torch.cat([event_vis, pred_vis, gt_vis], dim=4)
		vis = vis.detach()
		vis = torch.clamp(vis, 0, 255)
		vis = vis.to(torch.uint8)
		return vis
