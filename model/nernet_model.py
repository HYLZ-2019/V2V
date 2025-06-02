import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
# local modules

from .model_util import CropParameters, recursive_clone
from .base.base_model import BaseModel

from .nernet.representation_modules import Voxelization
from .nernet.unet import UNetNIAM_STcell_GCB, UNetRecurrent

def copy_states(states):
    """
    LSTM states: [(torch.tensor, torch.tensor), ...]
    GRU states: [torch.tensor, ...]
    """
    if states[0] is None:
        return copy.deepcopy(states)
    return recursive_clone(states)

class RepresentationRecurrent(BaseModel):
    """
    Compatible with E2VID_lightweight and Representation network
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.crop_size = unet_kwargs['crop_size']
        self.mlp_layers = unet_kwargs['mlp_layers']
        self.normalize = unet_kwargs['normalize']
        # self.normalize = False
        self.height = None
        self.width = None
        self.representation = None

        self.unet_kwargs = unet_kwargs

        self.network = unet_kwargs['recurrent_network']
        if self.network == 'NIAM_STcell_GCB':
            self.unetrecurrent = UNetNIAM_STcell_GCB(unet_kwargs)
        else:
            self.unetrecurrent = UNetRecurrent(unet_kwargs)

        self.set_resolution(256, 256) # Make placeholder so weights can be loaded in

    def set_resolution(self, H, W):
        if self.height is None or self.width is None:
            # First time setting resolution
            self.height = H
            self.width = W
            # Reset the resolution.
            device = next(self.unetrecurrent.parameters()).device
            self.representation = Voxelization(self.unet_kwargs, self.unet_kwargs['use_cnn_representation'], voxel_dimension=(self.num_bins, self.height, self.width), mlp_layers=self.mlp_layers, activation=nn.LeakyReLU(negative_slope=0.1), pretrained=True, normalize=self.normalize, combine_voxel=self.unet_kwargs['combine_voxel']).to(device)
            self.crop = CropParameters(self.width, self.height, self.num_encoders)
            return
        
        if H != self.height or W != self.width:
            # Resolution has changed. Keep the network parameters of Voxelization.
            old_state_dict = self.representation.state_dict()
            self.height = H
            self.width = W
            # Reset the resolution.
            device = next(self.unetrecurrent.parameters()).device
            self.representation = Voxelization(self.unet_kwargs, self.unet_kwargs['use_cnn_representation'], voxel_dimension=(self.num_bins, self.height, self.width), mlp_layers=self.mlp_layers, activation=nn.LeakyReLU(negative_slope=0.1), pretrained=True, normalize=self.normalize, combine_voxel=self.unet_kwargs['combine_voxel']).to(device)
            self.crop = CropParameters(self.width, self.height, self.num_encoders)
            # Copy the parameters from the old representation to the new one.
            self.representation.load_state_dict(old_state_dict, strict=True)
            return



    @property
    def states(self):
        return copy_states(self.unetrecurrent.states)

    @states.setter
    def states(self, states):
        self.unetrecurrent.states = states

    def reset_states(self):
        if 'NIAM' in self.network or 'NAS' in self.network:
            self.unetrecurrent.states = None
        else:
            self.unetrecurrent.states = [None] * self.unetrecurrent.num_encoders

    def forward(self, x):
        """
        :param x: events[x, y, t, p]
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        event_tensor = self.representation.forward(x)
        event_tensor = self.crop.pad(event_tensor)
        output_dict = self.unetrecurrent.forward(event_tensor)
        return output_dict, event_tensor