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

from model.train_utils import normalize_nobias, concat_imgs, inference_raft, load_raft

def flow2rgb_np(disp_x, disp_y, max_magnitude=None):
	# Input: (2, H, W) optical flow
	"""
	Convert an optic flow numpy array to an RGB color map for visualization.
	Operates directly on numpy arrays.

	:param disp_x: a [H x W] numpy array containing the X displacement
	:param disp_y: a [H x W] numpy array containing the Y displacement
	:param max_magnitude: optional maximum magnitude for normalization
	:returns rgb: a [H x W x 3] numpy array containing a color-coded representation of the flow [0, 255]
	"""
	assert disp_x.shape == disp_y.shape
	H, W = disp_x.shape

	# Calculate flow magnitude and angle
	magnitude = np.sqrt(disp_x**2 + disp_y**2)
	angle = np.arctan2(disp_y, disp_x)  # range [-pi, pi]

	# Convert angle to range [0, 1] for HSV hue
	h = (angle + np.pi) / (2 * np.pi)  # Map [-pi, pi] to [0, 2*pi], then divide by 2*pi

	# Normalize magnitude to range [0, 1] for HSV value
	if max_magnitude is None:
		max_mag = np.max(magnitude)
		# Avoid division by zero
		if max_mag == 0:
			max_mag = 1e-5
		v = magnitude / max_mag
	else:
		v = magnitude / max_magnitude
	v = np.clip(v, 0, 1) # Ensure value is in [0, 1]

	# Set saturation to 1
	s = np.ones_like(h)

	# --- HSV to RGB conversion ---
	# Based on https://en.wikipedia.org/wiki/HSL_and_HSV#HSV_to_RGB
	# and matching the PyTorch implementation logic

	# Calculate H' = H / 60 degrees. Since h is angle/(2*pi), H = h * 360. H' = H / 60 = h * 6.
	hp = h * 6.0  # H' in [0, 6]

	# Sector index i = floor(H')
	i = np.floor(hp).astype(int)
	# Fractional part f = H' - i
	f = hp - i

	# Intermediate values based on V and S
	p = v * (1.0 - s)
	q = v * (1.0 - s * f)
	t = v * (1.0 - s * (1.0 - f))

	# Initialize RGB channels
	r = np.zeros_like(h)
	g = np.zeros_like(h)
	b = np.zeros_like(h)

	# Select RGB values based on sector i
	# Handle i=6 for hp=6.0 edge case by modulo
	i = i % 6

	mask0 = (i == 0)
	mask1 = (i == 1)
	mask2 = (i == 2)
	mask3 = (i == 3)
	mask4 = (i == 4)
	mask5 = (i == 5)

	# Assign values based on masks
	# Sector 0: R=V, G=t, B=p
	r[mask0], g[mask0], b[mask0] = v[mask0], t[mask0], p[mask0]
	# Sector 1: R=q, G=V, B=p
	r[mask1], g[mask1], b[mask1] = q[mask1], v[mask1], p[mask1]
	# Sector 2: R=p, G=V, B=t
	r[mask2], g[mask2], b[mask2] = p[mask2], v[mask2], t[mask2]
	# Sector 3: R=p, G=q, B=V
	r[mask3], g[mask3], b[mask3] = p[mask3], q[mask3], v[mask3]
	# Sector 4: R=t, G=p, B=V
	r[mask4], g[mask4], b[mask4] = t[mask4], p[mask4], v[mask4]
	# Sector 5: R=V, G=p, B=q
	r[mask5], g[mask5], b[mask5] = v[mask5], p[mask5], q[mask5]

	# Stack channels and scale to [0, 255]
	rgb = np.stack((r, g, b), axis=-1) * 255.0
	return rgb.astype(np.uint8)



def flow2rgb(disp_x, disp_y, max_magnitude=None):
	"""
	Convert an optic flow tensor to an RGB color map for visualization
	Operates directly on torch tensors without numpy conversion

	:param disp_x: a [B x T x H x W] torch.Tensor containing the X displacement
	:param disp_y: a [B x T x H x W] torch.Tensor containing the Y displacement
	:param max_magnitude: optional maximum magnitude for normalization
	:returns rgb: a [B x T x 3 x H x W] torch.Tensor containing a color-coded representation of the flow [0, 255]
	"""
	assert(disp_x.shape == disp_y.shape)
	device = disp_x.device
	
	# Calculate flow magnitude and angle
	flows = torch.stack((disp_x, disp_y), dim=-1)
	magnitude = torch.norm(flows, dim=-1)
	
	# Convert angle to color (in degrees, range [0, 180])
	angle = torch.atan2(disp_y, disp_x)
	angle += torch.pi
	angle *= 180. / torch.pi / 2.
	# Keep angle as float for HSV calculations
	
	# Normalize magnitude to range [0, 255]
	if max_magnitude is None:
		max_mag = magnitude.flatten(start_dim=2).max(dim=-1)[0].view(disp_x.shape[0], disp_x.shape[1], 1, 1)
		# Avoid division by zero
		max_mag = torch.clamp(max_mag, min=1e-5)
		v = magnitude / max_mag * 255.0
	else:
		v = torch.clamp(255.0 * magnitude / max_magnitude, 0, 255)
	
	h = angle / 180.0
	s = torch.ones_like(h)
	
	# Claude 3.7 Sonnet Thinking failed to write the following code. It was produced by Gemini 2.5 Pro.

	# Convert HSV to RGB
	# Following the standard conversion formula
	# https://en.wikipedia.org/wiki/HSL_and_HSV#HSV_to_RGB
	# Scale V to [0, 1]
	v_norm = v / 255.0
	
	# Calculate H' = H / 60 degrees. Since h is angle/180, H' = h * 3.
	# Re-check: angle [0, 2pi] -> angle [0, 180] -> h [0, 1].
	# H should be in [0, 360]. H = h * 360. H' = H / 60 = h * 6.
	hp = h * 6.0  # H' in [0, 6]
	
	# Sector index i = floor(H')
	i = torch.floor(hp).long()
	# Fractional part f = H' - i
	f = hp - i.float()
	
	# Intermediate values based on V and S
	p = v_norm * (1.0 - s)
	q = v_norm * (1.0 - s * f)
	t_val = v_norm * (1.0 - s * (1.0 - f))
	
	# Initialize RGB channels
	r = torch.zeros_like(h)
	g = torch.zeros_like(h)
	b = torch.zeros_like(h)
	
	# Select RGB values based on sector i
	mask0 = (i == 0) | (i == 6) # Handle i=6 for hp=6.0 edge case
	mask1 = (i == 1)
	mask2 = (i == 2)
	mask3 = (i == 3)
	mask4 = (i == 4)
	mask5 = (i == 5)
	
	# Assign values based on masks
	# Sector 0: R=V, G=t, B=p
	r[mask0], g[mask0], b[mask0] = v_norm[mask0], t_val[mask0], p[mask0]
	# Sector 1: R=q, G=V, B=p
	r[mask1], g[mask1], b[mask1] = q[mask1], v_norm[mask1], p[mask1]
	# Sector 2: R=p, G=V, B=t
	r[mask2], g[mask2], b[mask2] = p[mask2], v_norm[mask2], t_val[mask2]
	# Sector 3: R=p, G=q, B=V
	r[mask3], g[mask3], b[mask3] = p[mask3], q[mask3], v_norm[mask3]
	# Sector 4: R=t, G=p, B=V
	r[mask4], g[mask4], b[mask4] = t_val[mask4], p[mask4], v_norm[mask4]
	# Sector 5: R=V, G=p, B=q
	r[mask5], g[mask5], b[mask5] = v_norm[mask5], p[mask5], q[mask5]
	
	# Stack channels and scale to [0, 255]
	# Input shape: [B, T, H, W] -> Output shape: [B, T, 3, H, W]
	rgb = torch.stack((r, g, b), dim=2) * 255.0
	return rgb

class FlowModelInterface():
	def __init__(self, configs, device, local_rank):
		
		self.configs = configs
		self.device = device
		self.local_rank = local_rank

		# Should be self.flow_model, but use variable name e2vid_model so it will be compatible with the e2vid training code in train_torch.py
		self.e2vid_model = instantiate_from_config(configs["model"]).to(device)

		# RAFT is used for generation of optical flow  ground truth.
		self.optical_flow_source = configs["loss"].get("optical_flow_source", "gt")
		assert self.optical_flow_source in ["raft_small", "raft_large", "gt", "zeros"], f"Unknown optical flow source: {self.optical_flow_source}"

		if self.optical_flow_source == "raft_small":
			self.raft_model = load_raft("raft_small", device=device)
		elif self.optical_flow_source == "raft_large":
			self.raft_model = load_raft("raft_large", device=device)

		self.raft_num_flow_updates = configs["loss"].get("raft_num_flow_updates", 12)

		self.loss_functions = self.load_loss_functions(configs["loss"])

		self.forward_type = configs.get("forward_type", "evflow")
		assert self.forward_type in ["evflow", "eraft"], f"Unsupported forward type: {self.forward_type}"

	def set_current_epoch(self, epoch):
		self.current_epoch = epoch
		
	def compute_metrics(self, pred, batch):
		# Calculate metrics for test.
		# Only use EPE. batch["flow"] may be ground truth (from MVSEC) or pseudo GT (from RAFT). In any case, just calculate the EPE.
		sequence_name = batch["sequence_name"][0]
		data_source_idx = batch["data_source_idx"][0]
		data_source = data_sources[data_source_idx] 
		sequence_name = batch["sequence_name"][0][0]
		log_prefix = f"{data_source.upper()}/{sequence_name}"

		B, Tp1, C, H, W = batch["frame"].shape
		T = Tp1 - 1 # For flow task, there is additional frame
		assert B == 1, "Batch size must be 1 for testing."
		
		metrics = defaultdict(list)

		for t in range(T):
			pred_flow = pred[0, t, :, :, :].detach()  # [2, H, W]
			gt_flow = batch["flow"][0, t, :, :, :].detach()  # [2, H, W]

			# Where gt_flow is not NaN over all two channels and both components are not zero
			gt_valid_mask = torch.logical_not(
				torch.isnan(gt_flow[0, :, :]) | torch.isnan(gt_flow[1, :, :]) | torch.logical_and(gt_flow[0, :, :] == 0, gt_flow[1, :, :] == 0)
			)
			
			events = batch["events"][0, t, :, :, :].detach()  # [C, H, W]
			events_mask = torch.abs(events).sum(axis=0) > 0
			sparse_mask = gt_valid_mask & events_mask
			
			# 2-norm over the two channels
			EE = ((pred_flow - gt_flow)**2).sum(axis=0) ** 0.5

			dense_pix_cnt = torch.sum(gt_valid_mask).item()
			sparse_pix_cnt = torch.sum(sparse_mask).item()
			total_pix_cnt = gt_valid_mask.numel()

			# I think it is much more reasonable to calculate average over all valid pixels. 
			reasonable = True

			# Calculate dense metrics over gt_valid_mask == 1
			# Set invalid pixels to 0, so they will not provide any EPE
			dense_EE = torch.where(gt_valid_mask, EE, torch.zeros_like(EE))
			div = dense_pix_cnt if reasonable else total_pix_cnt
			if dense_pix_cnt > 0:
				dense_EPE = torch.sum(dense_EE) / div
				dense_1PE = torch.sum((dense_EE > 1).float()) / div
				dense_3PE = torch.sum((dense_EE > 3).float()) / div
			else:
				dense_EPE = dense_1PE = dense_3PE = torch.tensor(0, device=self.device)

			# Calculate sparse metrics over gt_valid_mask & events_mask == 1
			sparse_EE = torch.where(sparse_mask, EE, torch.zeros_like(EE))
			div = sparse_pix_cnt if reasonable else total_pix_cnt
			if sparse_pix_cnt > 0:
				sparse_EPE = torch.sum(sparse_EE) / div
				sparse_1PE = torch.sum((sparse_EE > 1).float()) / div
				sparse_3PE = torch.sum((sparse_EE > 3).float()) / div
			else:
				sparse_EPE = sparse_1PE = sparse_3PE = torch.tensor(0, device=self.device)

			for metric_name, metric_value in zip(
				["dense_EPE", "dense_1PE", "dense_3PE", "sparse_EPE", "sparse_1PE", "sparse_3PE"],
				[dense_EPE, dense_1PE, dense_3PE, sparse_EPE, sparse_1PE, sparse_3PE]
			):
				metrics[f"{log_prefix}/{metric_name}"].append(metric_value.item())
			
		return metrics

	def load_loss_functions(self, cfgs):
		funcs = []
		if cfgs.get("l1_weight", 0) != 0:
			l1_weight = cfgs["l1_weight"]
			funcs.append(
				l1_loss(weight=l1_weight)
			)
		return funcs
	
	def forward_sequence(self, sequence, reset_states=True, test=False, val=False):
		# If there is no ground truth optical flow, predict some.
		# In testing, there may be ground truth flow, such as with MVSEC. Then we use GT. Else, we predict pseudo GT.
		if self.optical_flow_source == "gt" or (test == True and "flow" in sequence):
			assert "flow" in sequence
		
		elif self.optical_flow_source in ["raft_small", "raft_large"]:			
			# sequence["frame"]: B, (T+1), C, H, W
			img1 = sequence["frame"][:, :-1, :, :, :]
			img2 = sequence["frame"][:, 1:, :, :, :]
			flow = inference_raft(
				self.raft_model, 
				self.raft_num_flow_updates, 
				img1, img2
			)
			sequence["flow"] = flow

		elif self.optical_flow_source == "zeros":
			# Create a zero flow tensor
			B, Tp1, C, H, W = sequence["frame"].shape
			sequence["flow"] = torch.zeros((B, Tp1-1, 2, H, W), device=sequence["frame"].device)

		if reset_states:
			if self.local_rank is not None:
				self.e2vid_model.module.reset_states()
			else:
				self.e2vid_model.reset_states()

		# Real forward pass
		if self.forward_type == "evflow":
			return self.forward_sequence_evflow(sequence)
		elif self.forward_type == "eraft":
			return self.forward_sequence_eraft(sequence)
		
	def forward_sequence_evflow(self, sequence):

		B, T, C, H, W = sequence["events"].shape

		PAD = 16
		padded_h = int(np.ceil(H / PAD) * PAD)
		padded_w = int(np.ceil(W / PAD) * PAD)
		padded_events = torch.zeros((B, T, C, padded_h, padded_w), device=sequence["events"].device)
		padded_events[:,:,:,:H,:W] = sequence["events"]

		pred_flow = torch.zeros((B, T, 2, H, W), device=sequence["events"].device)
		
		for t in range(T):
			pred = self.e2vid_model(padded_events[:, t, :, :, :])
			pred_flow[:, t, :, :, :] = pred['flow'][:, :, :H, :W]
		
		# pred_imgs should be in [0, 1]
		return pred_flow
	
	def forward_sequence_eraft(self, sequence):
		# ERAFT provides padding and unpadding inside.
		# ERAFT takes voxels [t_{i-1}, t_i] and [t_i, t_{i+1}] as input, and outputs flow[t_i -> t_{i+1}].
		B, Tp1, C, H, W = sequence["events"].shape
		T = Tp1 - 1
		pred_flow = torch.zeros((B, T, 2, H, W), device=sequence["events"].device)

		for t in range(T):
			pred = self.e2vid_model(sequence["events"][:, t, :, :, :], sequence["events"][:, t+1, :, :, :])
			pred_flow[:, t, :, :, :] = pred
		
		return pred_flow

	def calc_loss(self, sequence, pred, remove_flow_loss=False):
		# remove_flow_loss is a dummy argument to be compatible with the e2vid code.

		# pred should be in [0, 1]
		B, T, C, H, W = sequence["flow"].shape
		losses = {}

		loss_functions_list = self.loss_functions

		for loss_ftn in loss_functions_list:
			loss_name = loss_ftn.__class__.__name__
			losses[loss_name] = torch.zeros((B, T), device=self.device)

		final_losses = collections.defaultdict(lambda: 0)
	
		for t in range(T):
			# Calculate the losses. Loss is [b], because sequences in one batch may be from different data sources.
			gt_flow = sequence["flow"][:, t, :, :, :]
			pred_flow = pred[:, t, :, :, :]
			
			for loss_ftn in loss_functions_list:
				loss_name = loss_ftn.__class__.__name__
				if isinstance(loss_ftn, l1_loss):
					ls = loss_ftn(pred_flow, gt_flow, reduce_batch=False)
				else:
					raise NotImplementedError(f"Unknown loss function: {loss_ftn}")
				losses[loss_name][:, t] = ls
		
		data_source_indices = sequence['data_source_idx']  # Shape is (B,)
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
		event_vis = normalize_nobias(torch.sum(batch["events"].detach(), dim=2, keepdim=True))*255 # B, T, 1, H, W
		Bf, Tf, Cf, Hf, Wf = batch["frame"].shape # When there is additional frame, Tf = T + 1
		gt_vis = batch["frame"][:, Tf-T:, :, :, :].detach()*255 # B, T, 1, H, W
		
		# Has additional voxel at beginning for ERAFT-style forwarding
		if event_vis.shape[1] > T:
			event_vis = event_vis[:, 1:, :, :, :]

		# Convert to 3 channels
		event_vis = event_vis.repeat(1, 1, 3, 1, 1)
		if gt_vis.shape[2] == 1:
			gt_vis = gt_vis.repeat(1, 1, 3, 1, 1)
		gt_vis = gt_vis.flip(dims=[2])  # Show RGB, not BGR

		# Calculate max flow magnitude from gt flow
		gt_flow = batch["flow"].detach()
		max_flow_magnitude = max(torch.norm(gt_flow, dim=2).max(), 0.1)

		pred_flow = pred.detach()

		pred_flow_vis = flow2rgb(pred_flow[:, :, 0, :, :], pred_flow[:, :, 1, :, :], max_magnitude=max_flow_magnitude)
		gt_flow_vis = flow2rgb(gt_flow[:, :, 0, :, :], gt_flow[:, :, 1, :, :], max_magnitude=max_flow_magnitude)

		flow_error = torch.abs(pred_flow - gt_flow).mean(axis=2, keepdim=True)
		flow_error_vis = torch.clip(flow_error / max_flow_magnitude, 0, 1).repeat(1, 1, 3, 1, 1) * 255

		vis = torch.cat([pred_flow_vis, gt_flow_vis, flow_error_vis, event_vis, gt_vis], dim=4)
		vis = vis.detach()
		vis = torch.clamp(vis, 0, 255)
		vis = vis.to(torch.uint8)
		return vis
