# The HyperE2VID version of E2VIDRecurrent.

import torch
from torch import nn
import copy
# These modules are the same to e2vid++ version
from model.submodules import ConvLayer, UpsampleConvLayer, TransposedConvLayer, RecurrentConvLayer, ResidualBlock
from model.hyper import ConvolutionalContextFusion, DynamicAtomGeneration, DynamicConv
from model.model_util import skip_sum, skip_concat

def recursive_clone(tensor):
    """
    Assumes tensor is a torch.tensor with 'clone()' method, possibly
    inside nested iterable.
    E.g., tensor = [(pytorch_tensor, pytorch_tensor), ...]
    """
    if hasattr(tensor, 'clone'):
        return tensor.clone()
    try:
        return type(tensor)(recursive_clone(t) for t in tensor)
    except TypeError:
        print('{} is not iterable and has no clone() method.'.format(tensor))

def copy_states(states):
    """
    LSTM states: [(torch.tensor, torch.tensor), ...]
    GRU states: [torch.tensor, ...]
    """
    if states[0] is None:
        return copy.deepcopy(states)
    return recursive_clone(states)

class DynamicUpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu',
                 in_fuse_channels=6, out_fuse_channels=32, num_atoms=6):
        super().__init__()

        # Convolutional context fusion:
        self.context_fusion = ConvolutionalContextFusion(in_fuse_channels, out_fuse_channels)

        self.dynamic_atom_generation = DynamicAtomGeneration(kernel_size=kernel_size, num_atoms=num_atoms, num_bases=6,
                                                             in_context_channels=out_fuse_channels, hid_channels=64,
                                                             stride=stride)

        self.dynamic_conv = DynamicConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding, num_atoms=num_atoms)

        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

    def forward(self, x, ev_tensor, prev_recs):
        x_upsampled = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        context = self.context_fusion(ev_tensor, prev_recs)
        dynamic_atoms = self.dynamic_atom_generation(context)
        out = self.dynamic_conv(x_upsampled, dynamic_atoms)
        if self.activation is not None:
            out = self.activation(out)
        return out

class BaseUNet(nn.Module):
    def __init__(self, base_num_channels, num_encoders, num_residual_blocks, num_output_channels, skip_type, norm,
                 use_upsample_conv, num_bins, recurrent_block_type=None, kernel_size=5, channel_multiplier=2,
                 use_dynamic_decoder=False):
        super().__init__()
        self.base_num_channels = base_num_channels
        self.num_encoders = num_encoders
        self.num_residual_blocks = num_residual_blocks
        self.num_output_channels = num_output_channels
        self.kernel_size = kernel_size
        self.skip_type = skip_type
        self.norm = norm
        self.num_bins = num_bins
        self.recurrent_block_type = recurrent_block_type
        self.use_dynamic_decoder = use_dynamic_decoder

        self.encoder_input_sizes = [int(self.base_num_channels * pow(channel_multiplier, i)) for i in
                                    range(self.num_encoders)]
        self.encoder_output_sizes = [int(self.base_num_channels * pow(channel_multiplier, i + 1)) for i in
                                     range(self.num_encoders)]
        self.max_num_channels = self.encoder_output_sizes[-1]

        self.skip_ftn = eval('skip_' + skip_type)

        if use_upsample_conv:
            self.UpsampleLayer = UpsampleConvLayer
        else:
            self.UpsampleLayer = TransposedConvLayer

        assert self.num_output_channels > 0

    def build_encoders(self):
        encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            encoders.append(RecurrentConvLayer(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                recurrent_block_type=self.recurrent_block_type, norm=self.norm))
        return encoders

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(zip(decoder_input_sizes, decoder_output_sizes)):

            if idx == 0 and self.use_dynamic_decoder:
                decoders.append(DynamicUpsampleLayer(
                    2 * input_size if self.skip_type == 'concat' else input_size,
                    output_size, kernel_size=self.kernel_size, padding=self.kernel_size // 2,
                    in_fuse_channels=1 + self.num_bins))
            else:
                decoders.append(self.UpsampleLayer(
                    2 * input_size if self.skip_type == 'concat' else input_size,
                    output_size, kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2, norm=self.norm))

        return decoders

    def build_prediction_layer(self, num_output_channels, norm=None):
        return ConvLayer(2 * self.base_num_channels if self.skip_type == 'concat' else self.base_num_channels,
                         num_output_channels, 1, padding="same", activation=None, norm=norm)

    def build_head_layer(self):
        head = ConvLayer(self.num_bins, self.base_num_channels,
                         kernel_size=self.kernel_size, stride=1,
                         padding=self.kernel_size // 2)  # N x C x H x W -> N x 32 x H x W

        return head


class UNetRecurrent(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, unet_kwargs):
        if 'num_output_channels' not in unet_kwargs:
            unet_kwargs['num_output_channels'] = 1
        final_activation = unet_kwargs.pop('final_activation', 'none')
        self.final_activation = getattr(torch, final_activation, None)
        super().__init__(**unet_kwargs)

        self.head = self.build_head_layer()
        self.encoders = self.build_encoders()
        self.build_resblocks()
        self.decoders = self.build_decoders()

        self.pred = self.build_prediction_layer(self.num_output_channels, self.norm)
        self.states = [None] * self.num_encoders

    def forward(self, x, prev_recs=None):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """
        # head
        ev_tensor = x
        x = self.head(x)

        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, self.states[i])
            blocks.append(x)
            self.states[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        for i, decoder in enumerate(self.decoders):
            skip_from_encoder = blocks[self.num_encoders - i - 1]
            if isinstance(decoder, DynamicUpsampleLayer):
                x = decoder(self.skip_ftn(x, skip_from_encoder), ev_tensor, prev_recs)
            else:
                x = decoder(self.skip_ftn(x, skip_from_encoder))

        img = self.pred(self.skip_ftn(x, head))
        if self.final_activation is not None:
            img = self.final_activation(img)

        if self.num_output_channels == 3:
            return {'image': img[:, 0:1, :, :], 'flow': img[:, 1:3, :, :]}
        elif self.num_output_channels == 1:
            return {'image': img}

class HyperE2VID(nn.Module):
    """
    Compatible with E2VID_lightweight
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """

    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.unetrecurrent = UNetRecurrent(unet_kwargs)
        self.prev_recs = None

    @property
    def states(self):
        return copy_states(self.unetrecurrent.states)

    @states.setter
    def states(self, states):
        self.unetrecurrent.states = states

    def reset_states(self):
        self.unetrecurrent.states = [None] * self.unetrecurrent.num_encoders
        self.prev_recs = None

    def forward(self, event_tensor, gt_image=None, beta=0):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        if self.prev_recs is None:
            self.prev_recs = torch.zeros(event_tensor.shape[0], 1, event_tensor.shape[2], event_tensor.shape[3], device=event_tensor.device)
        if gt_image is not None and beta > 0:
            prev_recs = self.prev_recs * (1 - beta) + gt_image * beta
        else:
            prev_recs = self.prev_recs
        output_dict = self.unetrecurrent.forward(event_tensor, prev_recs)
        self.prev_recs = output_dict['image'].detach()
        return output_dict
