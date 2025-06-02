import math
import torch
from torch import nn
import numpy as np

class PositionalEncodingSine(nn.Module):
    def __init__(self, d_hid, n_position=20000):
        super().__init__()
        pos_table = self._get_sinusoid_encoding_table(n_position, d_hid)
        self.register_buffer("pos_table", pos_table)

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.pos_table[:, :x.size(1)]

def build_position_encoding(pos_type, d_model):
    if pos_type == 'sine':
        position_embedding = PositionalEncodingSine(d_model)
    else:
        raise ValueError(f"not support {pos_type}")
    return position_embedding
