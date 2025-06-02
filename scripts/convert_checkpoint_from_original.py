# Run this script in the root directory of another codebase, such as https://github.com/TimoStoff/event_cnn_minimal, since the torch.load will try to import model classes from the original code structure. Then move the extracted state_dict-only checkpoint wherever you need it.

import torch

ckpt = torch.load("pretrained/flow_model.pth")

new_dict = {
	"state_dict": ckpt["state_dict"],
}

torch.save(new_dict, "pretrained/flow_model_state_dict.pth")

'''
# Code for ERAFT:

import torch

in_pth = "mvsec_20.tar"
checkpoint = torch.load(in_pth, map_location="cpu")
out_pth = "checkpoints/eraft_original/eraft_mvsec_20.pth"

new_dict = {
	"state_dict": checkpoint["model"],
}
torch.save(new_dict, out_pth)

'''