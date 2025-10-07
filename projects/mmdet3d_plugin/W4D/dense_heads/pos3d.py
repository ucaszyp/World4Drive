"""Modified Flash version of zoe model for fast training."""

import torch.utils.checkpoint
from torch import nn
import torchvision.transforms.functional as F
import numpy as np
import math



class Ego3DPositionEmbeddingMLP(nn.Module):
    """Absolute pos embedding, learned.
    https://github.com/kwea123/nerf_pl/blob/52aeb387da64a9ad9a0f914ea9b049ffc598b20c/models/nerf.py#L4
    """

    def __init__(self, in_channels=3, num_pos_feats=768, n_freqs=8, logscale=True):
        super(Ego3DPositionEmbeddingMLP, self).__init__()
        self.n_freqs = n_freqs
        self.freq_out_channels = in_channels * (2 * n_freqs + 1)
        if logscale:
            freq_bands = 2 ** torch.linspace(0, n_freqs - 1, n_freqs)
        else:
            freq_bands = torch.linspace(1, 2 ** (n_freqs - 1), n_freqs)
        
        center = torch.tensor([0., 0., 2.]).repeat(in_channels // 3)
        self.register_buffer("freq_bands", freq_bands, persistent=False)
        self.register_buffer("center", center, persistent=False)

        self.position_embedding_head = nn.Sequential(
            nn.Linear(self.freq_out_channels, num_pos_feats),
            nn.LayerNorm(num_pos_feats),
            nn.ReLU(),
            nn.Linear(num_pos_feats, num_pos_feats),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        """init with small weights to maintain stable training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.01)

    @torch.no_grad()
    def frequency_encoding(self, xyz):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        x \in [-2, 2]
        y \in [-2, 2]
        z \in [0., 4]
        Inputs:
            x: (b n m)
        Outputs:
            out: (b n o)
        """
        xyz_n = ((xyz - self.center) / 2.0).to(self.freq_bands.dtype)
        xyz_feq = xyz_n.unsqueeze(-1) * self.freq_bands  # (b n m 1)
        sin_xyz, cos_xyz = torch.sin(xyz_feq), torch.cos(xyz_feq)  # (b n m nf)
        encoding = torch.cat([xyz_n.unsqueeze(-1), sin_xyz, cos_xyz], -1).reshape(*xyz.shape[:2], -1)
        return encoding

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, N, F)."""
        # TODO: encoding with 3D position
        freq_encoding = self.frequency_encoding(xyz)
        position_embedding = self.position_embedding_head(freq_encoding)
        return position_embedding


def get_resize_output_image_size(
    input_height: int,
    input_width: int,
    output_size: tuple = (384, 512),
    keep_aspect_ratio: bool = True,
    multiple: int = 32,
):
    def constrain_to_multiple_of(val, multiple, min_val=0):
        x = (np.round(val / multiple) * multiple).astype(int)
        if x < min_val:
            x = math.ceil(val / multiple) * multiple
        return x

    output_height, output_width = output_size
    scale_height = output_height / input_height
    scale_width = output_width / input_width

    if keep_aspect_ratio:
        # scale as little as possible
        if abs(1 - scale_width) < abs(1 - scale_height):
            scale_height = scale_width
        else:
            scale_width = scale_height

    new_height = constrain_to_multiple_of(scale_height * input_height, multiple=multiple)
    new_width = constrain_to_multiple_of(scale_width * input_width, multiple=multiple)

    return (int(new_height), int(new_width))


def process_zoe(pixel_values, pad_mode="reflect", output_size=(384, 512)):
    """https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/zoedepth/image_processing_zoedepth.py"""
    # h, w = images.shape[-2:]
    # pad images
    ph, pw = 31, 31  # int((h / 2)**0.5 * 3), int((w / 2)**0.5 * 3) # 32, 31
    images = torch.nn.functional.pad(pixel_values, (pw, pw, ph, ph), mode=pad_mode)

    # resize images
    size = (384, 384)  # get_resize_output_image_size(h, w, output_size=output_size, keep_aspect_ratio=True, multiple=32) # 384, 384
    images = torch.nn.functional.interpolate(images, size=size, mode="bicubic", align_corners=True)

    # NOTE: zoe: padding -> resize -> nomalize.
    # BUT: siglip processor get nomalized image, we simplely follow `nomalize -> padding -> resize` in reflect pad mode
    ZOE_MEAN, ZOE_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    images = F.normalize(images, mean=ZOE_MEAN, std=ZOE_STD)
    return images, ph, pw