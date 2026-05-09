# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin
from diffusers.models.autoencoders.autoencoder_kl import (
    AutoencoderKLOutput,
    DecoderOutput,
    DiagonalGaussianDistribution,
)
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import logging
from diffusers.utils.accelerate_utils import apply_forward_hook
from einops import rearrange

_ACTS = {
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "mish": nn.Mish,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "identity": nn.Identity,
    "none": nn.Identity,
}


def resolve_activation(x):
    if x is None:
        return nn.Identity()
    if isinstance(x, nn.Module):
        return x
    name = str(x).strip().lower()
    if name in _ACTS:
        return _ACTS[name]()
    if name in ("lrelu", "leaky_relu"):
        return nn.LeakyReLU(0.01)
    raise ValueError(f"Unknown activation: {x}")


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

CACHE_T = 0
LATENT_T_STRIDE = 100
GRADIENT_CHECKPOINTING = False

class AvgDown3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            B,
            C * self.factor,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.mean(dim=2)
        return x


class DupUp3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk=False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class WanCausalConv3d(nn.Conv3d):
    r"""
    A custom 3D causal convolution layer with feature caching support.

    This layer extends the standard Conv3D layer by ensuring causality in the time dimension and handling feature
    caching for efficient inference.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to all three sides of the input. Default: 0
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Set up causal padding
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None, mode=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]

        if mode == 'upsample3d':
            # x: BCTHW
            assert self.stride[0] == 1 and self.stride[1] == 1 and self.stride[2] == 1
            assert self.kernel_size[0] == 3

            assert padding[0] == padding[1] and padding[2] == padding[3]

            results = []
            for i in range(x.shape[2] if padding[-2] == 2 else x.shape[2] - 1):
                if padding[-2] == 2:
                    if i == 0:
                        out = F.conv3d(x[:, :, 0:1, :, :], self.weight, self.bias, self.stride, (2, padding[2], padding[0]))[:, :, :-2]  # BC1HW
                    elif i == 1:
                        out = F.conv3d(x[:, :, 0:2, :, :], self.weight, self.bias, self.stride, (1, padding[2], padding[0]))[:, :, :-1]  # BC1HW
                    else:
                        out = F.conv3d(x[:, :, i - 2: i - 2 + self.kernel_size[0], :, :], self.weight, self.bias, self.stride, (0, padding[2], padding[0]))  # BC1HW
                elif padding[-2] == 1:
                    if i == 0:
                        out = F.conv3d(x[:, :, 0:2, :, :], self.weight, self.bias, self.stride, (1, padding[2], padding[0]))[:, :, :-1]  # BC1HW
                    else:
                        out = F.conv3d(x[:, :, i - 1: i - 1 + self.kernel_size[0], :, :], self.weight, self.bias, self.stride, (0, padding[2], padding[0]))  # BC1HW
                else:
                    raise ValueError("Invalid padding for causal conv3d in upsample3d mode.")
                results.append(out)

            if not results:
                breakpoint() # TODO

            return torch.cat(results, dim=2)  # BCTHW

        x = F.pad(x, padding)
        return super().forward(x)


        '''
        if mode == "upsample3d":
            padding = list(self._padding)
            x = F.pad(x, padding)
            t = x.shape[2]
            itr = t - 2
            print(f"DEBUG: time frame {t}")
            out = super().forward(x[:, :, :1, :, :])
            for i in range(1, itr):
                out_ = super().forward(x[:, :, i: i + 4, :, :])
                out = torch.cat([out, out_], 2)
            return out
        else:
            padding = list(self._padding)
            if cache_x is not None and self._padding[4] > 0:
                cache_x = cache_x.to(x.device)
                x = torch.cat([cache_x, x], dim=2)
                padding[4] -= cache_x.shape[2]
            x = F.pad(x, padding)
            
            print(x.shape, self.weight.shape)
            print(x.dtype, self.weight.dtype)
            return super().forward(x)
        '''


class WanRMS_norm(nn.Module):
    r"""
    A custom RMS normalization layer.

    Args:
        dim (int): The number of dimensions to normalize over.
        channel_first (bool, optional): Whether the input tensor has channels as the first dimension.
            Default is True.
        images (bool, optional): Whether the input represents image data. Default is True.
        bias (bool, optional): Whether to include a learnable bias term. Default is False.
    """

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class WanUpsample(nn.Upsample):
    r"""
    Perform upsampling while ensuring the output tensor has the same data type as the input.

    Args:
        x (torch.Tensor): Input tensor to be upsampled.

    Returns:
        torch.Tensor: Upsampled tensor with the same data type as the input.
    """

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class WanResample(nn.Module):
    r"""
    A custom resampling module for 2D and 3D data.

    Args:
        dim (int): The number of input/output channels.
        mode (str): The resampling mode. Must be one of:
            - 'none': No resampling (identity operation).
            - 'upsample2d': 2D upsampling with nearest-exact interpolation and convolution.
            - 'upsample3d': 3D upsampling with nearest-exact interpolation, convolution, and causal 3D convolution.
            - 'downsample2d': 2D downsampling with zero-padding and convolution.
            - 'downsample3d': 3D downsampling with zero-padding, convolution, and causal 3D convolution.
    """

    def __init__(self, dim: int, mode: str, upsample_out_dim: int = None) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        # default to dim //2
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
            self.time_conv = WanCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = WanCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0], is_reference=False, first_chunk=False):
        b, c, t, h, w = x.size()

        if self.mode == "upsample3d":
            if feat_cache is not None and not is_reference:
                # Latent frames: full caching logic
                idx = feat_idx[0]

                if feat_cache[idx] is None:
                    if t <= 1:
                        feat_cache[idx] = "Rep"
                        feat_idx[0] += 1
                    else:
                        subseq = x[:, :, 1:]
                        cache_x = subseq[:, :, -CACHE_T:, :, :].clone() if CACHE_T > 0 else subseq[:, :, :0, :, :]
                        if cache_x.shape[2] < 2:
                            cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)

                        subseq = self.time_conv(subseq, mode=self.mode)

                        feat_cache[idx] = cache_x
                        feat_idx[0] += 1

                        subseq = subseq.reshape(b, 2, c, t - 1, h, w)
                        subseq = torch.stack((subseq[:, 0, :, :, :, :], subseq[:, 1, :, :, :, :]), 3)
                        subseq = subseq.reshape(b, c, (t - 1) * 2, h, w)
                        x = torch.cat([x[:, :, :1, :, :], subseq], dim=2)
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone() if CACHE_T > 0 else x[:, :, :0, :, :]
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)

                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x, mode=self.mode)
                    else:
                        x = self.time_conv(x, feat_cache[idx], mode=self.mode)

                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)

        # Spatial resampling (applies to all paths)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None and not is_reference:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    if t <= 1:
                        feat_cache[idx] = x.clone()
                        feat_idx[0] += 1
                    else:
                        subseq = x[:, :, 1:]
                        cache_x = subseq[:, :, -1:, :, :].clone()
                        subseq = self.time_conv(x)
                        x = torch.cat([x[:, :, :1, :, :], subseq], dim=2)
                        feat_cache[idx] = cache_x
                        feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class WanResidualBlock(nn.Module):
    r"""
    A custom residual block module.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        dropout (float, optional): Dropout rate for the dropout layer. Default is 0.0.
        non_linearity (str, optional): Type of non-linearity to use. Default is "silu".
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = resolve_activation(non_linearity)

        # layers
        self.norm1 = WanRMS_norm(in_dim, images=False)
        self.conv1 = WanCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = WanRMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = WanCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = WanCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # Apply shortcut connection
        h = self.conv_shortcut(x)

        # First normalization and activation
        x = self.norm1(x)
        x = self.nonlinearity(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone() if CACHE_T > 0 else x[:, :, :0, :, :]
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)

            x = self.conv1(x, feat_cache[idx], mode='upsample3d')
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x, mode='upsample3d')

        # Second normalization and activation
        x = self.norm2(x)
        x = self.nonlinearity(x)

        # Dropout
        x = self.dropout(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone() if CACHE_T > 0 else x[:, :, :0, :, :]
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)

            x = self.conv2(x, feat_cache[idx], mode='upsample3d')
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x, mode='upsample3d')

        # Add residual connection
        return x + h

class WanAttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.

    Args:
        dim (int): The number of channels in the input tensor.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = WanRMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        batch_size, channels, time, height, width = x.size()

        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)

        # compute query, key, value
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(q, k, v)

        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)

        # output projection
        x = self.proj(x)

        # Reshape back: [(b*t), c, h, w] -> [b, c, t, h, w]
        x = x.view(batch_size, time, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)

        return x + identity


class WanMidBlock(nn.Module):
    """
    Middle block for WanVAE encoder and decoder.

    Args:
        dim (int): Number of input/output channels.
        dropout (float): Dropout rate.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(self, dim: int, dropout: float = 0.0, non_linearity: str = "silu", num_layers: int = 1):
        super().__init__()
        self.dim = dim

        # Create the components
        resnets = [WanResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(WanAttentionBlock(dim))
            resnets.append(WanResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = GRADIENT_CHECKPOINTING

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # First residual block
        x = self.resnets[0](x, feat_cache, feat_idx)

        # Process through attention and residual blocks
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                if self.gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        attn,
                        x,
                        use_reentrant=False,
                    )
                else:
                    x = attn(x)
                    
            if self.gradient_checkpointing and feat_cache is not None:
                # Save mutable state before checkpoint; it will be restored on recompute.
                initial_idx = feat_idx[0]
                initial_cache_snapshot = [
                    (c.clone() if isinstance(c, torch.Tensor) else c)
                    for c in feat_cache
                ]

                def checkpoint_fn(x, block=resnet):
                    feat_idx[0] = initial_idx
                    for j in range(len(feat_cache)):
                        val = initial_cache_snapshot[j]
                        feat_cache[j] = val.clone() if isinstance(val, torch.Tensor) else val
                    return block(x, feat_cache, feat_idx)

                x = torch.utils.checkpoint.checkpoint(
                    checkpoint_fn,
                    x,
                    use_reentrant=False,
                )
            else:
                x = resnet(x, feat_cache, feat_idx)

        return x


class WanResidualDownBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, num_res_blocks, temperal_downsample=False, down_flag=False):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(WanResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = WanResample(out_dim, mode=mode)
        else:
            self.downsampler = None

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x_copy = x.clone()
        for resnet in self.resnets:
            x = resnet(x, feat_cache, feat_idx)
        if self.downsampler is not None:
            x = self.downsampler(x, feat_cache, feat_idx)

        return x + self.avg_shortcut(x_copy)


class WanEncoder3d(nn.Module):
    r"""
    A 3D encoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temperal_downsample (list of bool): Whether to downsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        non_linearity: str = "silu",
        is_residual: bool = False,  # wan 2.2 vae use a residual downblock
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.nonlinearity = resolve_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv_in = WanCausalConv3d(in_channels, dims[0], 3, padding=1)

        # downsample blocks
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if is_residual:
                self.down_blocks.append(
                    WanResidualDownBlock(
                        in_dim,
                        out_dim,
                        dropout,
                        num_res_blocks,
                        temperal_downsample=temperal_downsample[i] if i != len(dim_mult) - 1 else False,
                        down_flag=i != len(dim_mult) - 1,
                    )
                )
            else:
                for _ in range(num_res_blocks):
                    self.down_blocks.append(WanResidualBlock(in_dim, out_dim, dropout))
                    if scale in attn_scales:
                        self.down_blocks.append(WanAttentionBlock(out_dim))
                    in_dim = out_dim

                # downsample block
                if i != len(dim_mult) - 1:
                    mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                    self.down_blocks.append(WanResample(out_dim, mode=mode))
                    scale /= 2.0

        # middle blocks
        self.mid_block = WanMidBlock(out_dim, dropout, non_linearity, num_layers=1)
        
        # output blocks
        self.norm_out = WanRMS_norm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, z_dim, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone() if CACHE_T > 0 else x[:, :, :0, :, :]
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        ## downsamples
        for layer in self.down_blocks:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        x = self.mid_block(x, feat_cache, feat_idx)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone() if CACHE_T > 0 else x[:, :, :0, :, :]
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class WanResidualUpBlock(nn.Module):
    """
    A block that handles upsampling for the WanVAE decoder.

    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        temperal_upsample (bool): Whether to upsample on temporal dimension
        up_flag (bool): Whether to upsample or not
        non_linearity (str): Type of non-linearity to use
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        temperal_upsample: bool = False,
        up_flag: bool = False,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2,
            )
        else:
            self.avg_shortcut = None

        # create residual blocks
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        if up_flag:
            upsample_mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler = WanResample(out_dim, mode=upsample_mode, upsample_out_dim=out_dim)
        else:
            self.upsampler = None

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False, is_reference=False):
        """
        Forward pass through the upsampling block.

        Args:
            x (torch.Tensor): Input tensor
            feat_cache (list, optional): Feature cache for causal convolutions
            feat_idx (list, optional): Feature index for cache management
            first_chunk (bool, optional): Whether this is the first chunk
            is_reference (bool, optional): Whether processing reference tokens

        Returns:
            torch.Tensor: Output tensor
        """
        x_copy = x.clone()

        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache, feat_idx, is_reference=is_reference)
            else:
                x = resnet(x)

        if self.upsampler is not None:
            if feat_cache is not None:
                x = self.upsampler(x, feat_cache, feat_idx)
            else:
                # Pass is_reference to upsampler
                x = self.upsampler(x, is_reference=is_reference)

        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_copy, first_chunk=first_chunk, is_reference=is_reference)

        return x


class WanUpBlock(nn.Module):
    """
    A block that handles upsampling for the WanVAE decoder.

    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        upsample_mode (str, optional): Mode for upsampling ('upsample2d' or 'upsample3d')
        non_linearity (str): Type of non-linearity to use
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: Optional[str] = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Create layers list
        resnets = []
        # Add residual blocks and attention if needed
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([WanResample(out_dim, mode=upsample_mode)])

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=None, is_reference=False):
        """
        Forward pass through the upsampling block.

        Args:
            x (torch.Tensor): Input tensor
            feat_cache (list, optional): Feature cache for causal convolutions
            feat_idx (list, optional): Feature index for cache management
            first_chunk (bool, optional): Whether this is the first chunk
            is_reference (bool, optional): Whether processing reference tokens

        Returns:
            torch.Tensor: Output tensor
        """
        # Pass is_reference to all resnets
        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache, feat_idx)
            else:
                x = resnet(x)

        # Pass is_reference to upsampler
        if self.upsamplers is not None:
            if feat_cache is not None:
                x = self.upsamplers[0](x, feat_cache, feat_idx)
            else:
                x = self.upsamplers[0](x, first_chunk=first_chunk, is_reference=is_reference)
        return x


class RefConvIn(nn.Module):
    """
    Tokenizes reference videos by converting spatial resolution into channels.
    Uses only reshape operations.
    Converts [b, c, T, h, w] to [b, c_out, T, h/patch_size, w/patch_size]
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=384,
        patch_size=8,
    ):
        """
        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB)
            out_channels (int): Number of output channels
            patch_size (int): Size of spatial patches for downsampling
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size

        # Calculate intermediate channels after patchification
        self.patch_channels = in_channels * patch_size * patch_size

        # Conv2d layer to project from patch_channels to out_channels
        self.proj = nn.Conv2d(self.patch_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = WanRMS_norm(self.out_channels, images=True)

        # Calculate how many times to repeat
        assert (
            self.out_channels % self.patch_channels == 0
        ), f"out_channels ({self.out_channels}) must be divisible by patch_channels ({self.patch_channels})"


    def forward(self, x):
        """
        Tokenize reference input using only reshape operations.

        Args:
            x: Input tensor [b, in_channels, T, h, w]

        Returns:
            Tokenized tensor [b, out_channels, T, h/patch_size, w/patch_size]
        """
        b, c, T, h, w = x.shape
        patch_size = self.patch_size

        # Ensure dimensions are divisible by patch_size
        assert h % patch_size == 0, f"Height {h} must be divisible by patch_size {patch_size}"
        assert w % patch_size == 0, f"Width {w} must be divisible by patch_size {patch_size}"

        # Step 1: Reshape into patches
        x = x.view(b, c, T, h // patch_size, patch_size, w // patch_size, patch_size)

        # Step 2: Rearrange dimensions
        x = x.permute(0, 1, 4, 6, 2, 3, 5).contiguous()

        # Step 3: Flatten patches into channels
        x = x.view(b, c * patch_size * patch_size, T, h // patch_size, w // patch_size)

        # Step 4: Apply Conv2d projection for each time step
        # Reshape to merge batch and time dimensions
        x = x.view(b * T, self.patch_channels, h // patch_size, w // patch_size)

        # Apply convolution
        x = self.proj(x)
        x = self.norm(x)

        # Reshape back to separate batch and time dimensions
        x = x.view(b, self.out_channels, T, h // patch_size, w // patch_size)

        return x


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [
            self.attention_head_dim - 2 * (self.attention_head_dim // 3),
            self.attention_head_dim // 3,
            self.attention_head_dim // 3,
        ]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return freqs_cos, freqs_sin

class ReferenceRemover:
    """
    Removes reference frame tokens that were concatenated along temporal dimension.
    Handles cases where temporal upsampling may have occurred.
    """

    def __init__(self, ref_frame_count: int = 1):
        """
        Args:
            ref_frame_count: Number of reference frames concatenated (default: 1)
        """
        self.ref_frame_count = ref_frame_count

    def __call__(self, x: torch.Tensor, original_temporal_dim: int) -> torch.Tensor:
        """
        Remove reference frames from the temporal dimension.

        Args:
            x: Tensor of shape [B, C, T, H, W]
            original_temporal_dim: The temporal dimension before concatenating reference

        Returns:
            Tensor with reference frames removed
        """
        current_temporal_dim = x.shape[2]

        # Calculate temporal scale factor from upsampling
        original_input_frames = original_temporal_dim + 1
        temporal_scale = current_temporal_dim // original_input_frames

        # Calculate how many frames to remove (scaled reference frames)
        frames_to_remove = self.ref_frame_count * temporal_scale

        # Remove reference frames from the beginning
        return (x[:, :, :frames_to_remove, :, :], x[:, :, frames_to_remove:, :, :])


class WanDecoder3d(nn.Module):
    r"""
    A 3D decoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temperal_upsample (list of bool): Whether to upsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
        non_linearity (str): Type of non-linearity to use.
        skip_decoder_attention (bool): If True, skip all attention blocks in decoder.
    """

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
        non_linearity: str = "silu",
        out_channels: int = 3,
        is_residual: bool = False,
        use_reference: bool = False,
        skip_decoder_attention: bool = False,
        dc_factor: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample
        self.use_reference = use_reference
        self.skip_decoder_attention = skip_decoder_attention
        self.dc_factor = dc_factor
        self.nonlinearity = resolve_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        # init block
        self.conv_in = WanCausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.mid_block = WanMidBlock(dims[0], dropout, non_linearity, num_layers=1) 

        self.ref_conv_in = RefConvIn(out_channels=dims[0]) if self.use_reference else None

        # upsample block & attention block 1, 2 and 3
        self.up_blocks = nn.ModuleList([])

        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i > 0 and not is_residual:
                # wan vae 2.1
                in_dim = in_dim // 2

            # determine if we need upsampling
            up_flag = i != len(dim_mult) - 1
            # determine upsampling mode, if not upsampling, set to None
            upsample_mode = None
            if up_flag and temperal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"
            # Create and add the upsampling block
            if is_residual:
                up_block = WanResidualUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    temperal_upsample=temperal_upsample[i] if up_flag else False,
                    up_flag=up_flag,
                    non_linearity=non_linearity,
                )
            else:
                up_block = WanUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                    non_linearity=non_linearity,
                )

            self.up_blocks.append(up_block)

        # output blocks
        self.norm_out = WanRMS_norm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, out_channels, 3, padding=1)

        self.gradient_checkpointing = GRADIENT_CHECKPOINTING

    def forward(self, x, transformer, feat_cache=None, feat_idx=[0], first_chunk=False, reference_frame=None, skip=False, window_size=-1):
        run_attn = not self.skip_decoder_attention and not skip
        if self.gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                self.conv_in, 
                x,
                use_reentrant=False
                )
        else:
            x = self.conv_in(x)

        ## middle
        x = self.mid_block(x, feat_cache, feat_idx)
        ref_tokens = None
        if self.use_reference and reference_frame is not None:
            # ref_tokens: [B, C, 1, H, W] - single frame
            if self.gradient_checkpointing:
                ref_tokens = torch.utils.checkpoint.checkpoint(
                    self.ref_conv_in, 
                    reference_frame, 
                    use_reentrant=False
                )
            else:
                ref_tokens = self.ref_conv_in(reference_frame)

        # Transformer + upblock
        if run_attn:
            for i in range(4):
                if i <= 2:
                    if ref_tokens is not None:
                        x = torch.cat([ref_tokens, x], dim=2)
                    transformer_output = transformer(
                        hidden_states=x,
                        stage_idx=i,
                        return_dict=True,
                        window_size=window_size,
                    )
                    # Extract the output sample
                    x = transformer_output.sample if hasattr(transformer_output, 'sample') else transformer_output[0]
                    if ref_tokens is not None:
                        ref_tokens, x = x[:, :, :1], x[:, :, 1:]
                        if i <= 1:
                            if self.gradient_checkpointing:
                                ref_tokens = torch.utils.checkpoint.checkpoint(
                                    self.up_blocks[i], 
                                    ref_tokens,
                                    None,
                                    [0],
                                    first_chunk,
                                    True,
                                    use_reentrant=False
                                    )
                            else:
                                ref_tokens = self.up_blocks[i](ref_tokens, is_reference=True, first_chunk=first_chunk)
            
                if self.gradient_checkpointing:
                    # Save mutable state before checkpoint - will be restored on each forward run
                    # (both original forward and backward recompute)
                    initial_idx = feat_idx[0]
                    initial_cache_snapshot = [
                        (c.clone() if isinstance(c, torch.Tensor) else c)
                        for c in feat_cache
                    ] if feat_cache is not None else None

                    def checkpoint_fn(x, block_idx=i):
                        # Restore state before each run to ensure consistency
                        feat_idx[0] = initial_idx
                        if initial_cache_snapshot is not None:
                            for j in range(len(feat_cache)):
                                val = initial_cache_snapshot[j]
                                feat_cache[j] = val.clone() if isinstance(val, torch.Tensor) else val
                        return self.up_blocks[block_idx](x, feat_cache, feat_idx, first_chunk=first_chunk)

                    x = torch.utils.checkpoint.checkpoint(
                        checkpoint_fn,
                        x,
                        use_reentrant=False,
                    )
                else:
                    x = self.up_blocks[i](x, feat_cache, feat_idx, first_chunk=first_chunk)
        else:
            print(f"[DEBUG]: Transformer skipped")
            for i in range(4):
                x = self.up_blocks[i](x, feat_cache, feat_idx, first_chunk=first_chunk)
        
        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        
        if self.gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                self.conv_out,
                x,
                None,
                'upsample3d',
                use_reentrant=False,
            )
        else:
            x = self.conv_out(x, mode='upsample3d')
        return x


def patchify(x, patch_size):
    if patch_size == 1:
        return x

    if x.dim() != 5:
        raise ValueError(f"Invalid input shape: {x.shape}")
    # x shape: [batch_size, channels, frames, height, width]
    batch_size, channels, frames, height, width = x.shape

    # Ensure height and width are divisible by patch_size
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"Height ({height}) and width ({width}) must be divisible by patch_size ({patch_size})")

    # Reshape to [batch_size, channels, frames, height//patch_size, patch_size, width//patch_size, patch_size]
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size, width // patch_size, patch_size)

    # Rearrange to [batch_size, channels * patch_size * patch_size, frames, height//patch_size, width//patch_size]
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames, height // patch_size, width // patch_size)

    return x


def unpatchify(x, patch_size):
    if patch_size == 1:
        return x

    if x.dim() != 5:
        raise ValueError(f"Invalid input shape: {x.shape}")
    # x shape: [batch_size, (channels * patch_size * patch_size), frame, height, width]
    batch_size, c_patches, frames, height, width = x.shape
    channels = c_patches // (patch_size * patch_size)

    # Reshape to [b, c, patch_size, patch_size, f, h, w]
    x = x.view(batch_size, channels, patch_size, patch_size, frames, height, width)

    # Rearrange to [b, c, f, h * patch_size, w * patch_size]
    x = x.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
    x = x.view(batch_size, channels, frames, height * patch_size, width * patch_size)

    return x


class AutoencoderKLWan(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.
    Introduced in [Wan 2.1].

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        base_dim: int = 96,
        decoder_base_dim: Optional[int] = None,
        use_reference: bool = False,
        skip_decoder_attention: bool = False,
        z_dim: int = 16,
        dim_mult: Tuple[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[float] = [],
        temperal_downsample: List[bool] = [False, True, True],
        dropout: float = 0.0,
        latents_mean: List[float] = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ],
        latents_std: List[float] = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ],
        is_residual: bool = False,
        in_channels: int = 3,
        out_channels: int = 3,
        patch_size: Optional[int] = None,
        scale_factor_temporal: Optional[int] = 4,
        scale_factor_spatial: Optional[int] = 8,
        inference_w_dropout=False,
        dropout_p=0.7,
        gradient_checkpointing=False,
        **kwargs,
    ) -> None:
        global GRADIENT_CHECKPOINTING
        GRADIENT_CHECKPOINTING = gradient_checkpointing
        super().__init__()
        self.inference_w_dropout = inference_w_dropout
        self.dropout_p = dropout_p

        self.z_dim = z_dim
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        if decoder_base_dim is None:
            decoder_base_dim = base_dim

        self.encoder = WanEncoder3d(
            in_channels=in_channels,
            dim=base_dim,
            z_dim=z_dim * 2,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            dropout=dropout,
            is_residual=is_residual,
        )
        self.quant_conv = WanCausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = WanCausalConv3d(z_dim, z_dim, 1)

        self.decoder = WanDecoder3d(
            dim=decoder_base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_upsample=self.temperal_upsample,
            dropout=dropout,
            out_channels=out_channels,
            is_residual=is_residual,
            use_reference=use_reference,
            skip_decoder_attention=skip_decoder_attention,
        )

        self.spatial_compression_ratio = 2 ** len(self.temperal_downsample)

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
        # to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
        # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
        # intermediate tiles together, the memory requirement can be lowered.
        self.use_tiling = False

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192

        # Precompute and cache conv counts for encoder and decoder for clear_cache speedup
        self._cached_conv_counts = {
            "decoder": (
                sum(isinstance(m, WanCausalConv3d) for m in self.decoder.modules()) if self.decoder is not None else 0
            ),
            "encoder": (
                sum(isinstance(m, WanCausalConv3d) for m in self.encoder.modules()) if self.encoder is not None else 0
            ),
        }

        self.reference_frame = None

    def _init_ref_conv_in(self):
        ref_conv_in = getattr(self.decoder, "ref_conv_in", None)
        if ref_conv_in is None:
            return
        
        with torch.no_grad():
            nn.init.xavier_uniform_(ref_conv_in.proj.weight)
            if ref_conv_in.proj.bias is not None:
                nn.init.constant_(ref_conv_in.proj.bias, 0.0)

    def _apply_token_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply token dropout to the input tensor.

        Args:
            x: Input tensor of shape [B, C, T, H, W]

        Returns:
            Tensor with random tokens dropped (set to zero)
        """
        if self.inference_w_dropout or self.training:
            if self.training:
                p = torch.rand(1).item() * self.dropout_p
            else:
                p = self.dropout_p
            dropped = torch.rand_like(x[:, :1, :1, :, :]) < p
            x = torch.where(dropped, torch.zeros_like(x), x)
        return x

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_sample_stride_height: Optional[float] = None,
        tile_sample_stride_width: Optional[float] = None,
    ) -> None:
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.

        Args:
            tile_sample_min_height (`int`, *optional*):
                The minimum height required for a sample to be separated into tiles across the height dimension.
            tile_sample_min_width (`int`, *optional*):
                The minimum width required for a sample to be separated into tiles across the width dimension.
            tile_sample_stride_height (`int`, *optional*):
                The minimum amount of overlap between two consecutive vertical tiles. This is to ensure that there are
                no tiling artifacts produced across the height dimension.
            tile_sample_stride_width (`int`, *optional*):
                The stride between two consecutive horizontal tiles. This is to ensure that there are no tiling
                artifacts produced across the width dimension.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width

    def disable_tiling(self) -> None:
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_tiling = False

    def enable_slicing(self) -> None:
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self) -> None:
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def clear_cache(self):
        # Use cached conv counts for decoder and encoder to avoid re-iterating modules each call
        self._conv_num = self._cached_conv_counts["decoder"]
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = self._cached_conv_counts["encoder"]
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def _encode(self, x: torch.Tensor):
        _, _, num_frame, height, width = x.shape

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x, is_reference)

        self.clear_cache()
        if self.config.patch_size is not None:
            x = patchify(x, patch_size=self.config.patch_size)
        iter_ = 1 #TODO
        for i in range(0, iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, : 4 * LATENT_T_STRIDE - 3, :, :], 
                    feat_cache=self._enc_feat_map, 
                    feat_idx=self._enc_conv_idx,
                )
            else:
                out_ = self.encoder(
                    x[:, :, i * 4 * LATENT_T_STRIDE - 3 :  (i + 1) * 4 * LATENT_T_STRIDE - 3, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)

        enc = self.quant_conv(out)
        self.clear_cache()
        return enc

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        r"""
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded videos. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)

        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor, transformer, return_dict: bool = True, reference_frame=None, skip=False, window_size=-1):
        _, _, num_frame, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio

        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z, return_dict=return_dict, reference_frame=reference_frame, skip=skip)

        self.clear_cache()

        x = self.post_quant_conv(z)

        x = self._apply_token_dropout(x)

        for i in range(0, num_frame, LATENT_T_STRIDE):
            self._conv_idx = [0]
            self._conv_idx_ref = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i : i + LATENT_T_STRIDE, :, :],
                    transformer=transformer,          
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=True,
                    reference_frame=reference_frame,
                    skip=skip,
                    window_size=window_size,
                )
            else:
                out_ = self.decoder(
                    x[:, :, i : i + LATENT_T_STRIDE, :, :],
                    transformer=transformer,
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    reference_frame=reference_frame,
                    skip=skip,
                    window_size=window_size,
                )
                out = torch.cat([out, out_], 2)

        if self.config.patch_size is not None:
            out = unpatchify(out, patch_size=self.config.patch_size)

        out = torch.clamp(out, min=-1.0, max=1.0)

        self.clear_cache()
        if not return_dict:
            return (out,)

        return DecoderOutput(sample=out)

    @apply_forward_hook
    def decode(
        self, z: torch.Tensor, transformer ,return_dict: bool = True, reference_frame=None, skip=False, window_size=-1
    ) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.
            reference_frame (`torch.Tensor`, *optional*):
                Reference frame for decoder attention.
            skip (`bool`, *optional*, defaults to `False`):
                Whether to skip attention in the decoder.
        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        # Use passed reference_frame or fall back to stored one
        ref_frame = reference_frame if reference_frame is not None else self.reference_frame

        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [
                self._decode(z_slice, transformer, reference_frame=ref_frame, skip=skip, window_size=window_size).sample for z_slice in z.split(1)
            ]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z, transformer, reference_frame=ref_frame, skip=skip, window_size=window_size).sample

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)
    
    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def tiled_encode(self, x: torch.Tensor) -> AutoencoderKLOutput:
        r"""Encode a batch of images using a tiled encoder.

        Args:
            x (`torch.Tensor`): Input batch of videos.

        Returns:
            `torch.Tensor`:
                The latent representation of the encoded videos.
        """
        _, _, num_frames, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        # Split x into overlapping tiles and encode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, self.tile_sample_stride_height):
            row = []
            for j in range(0, width, self.tile_sample_stride_width):
                self.clear_cache()
                time = []
                frame_range = 1 + (num_frames - 1) // 4
                for k in range(frame_range):
                    self._enc_conv_idx = [0]
                    if k == 0:
                        tile = x[:, :, :1, i : i + self.tile_sample_min_height, j : j + self.tile_sample_min_width]
                    else:
                        tile = x[
                            :,
                            :,
                            1 + 4 * (k - 1) : 1 + 4 * k,
                            i : i + self.tile_sample_min_height,
                            j : j + self.tile_sample_min_width,
                        ]
                    tile = self.encoder(tile, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
                    tile = self.quant_conv(tile)
                    time.append(tile)
                row.append(torch.cat(time, dim=2))
            rows.append(row)
        self.clear_cache()

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))

        enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
        return enc

    def tiled_decode(
        self, z: torch.Tensor, return_dict: bool = True, reference_frame=None, skip=False
    ) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        _, _, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                self.clear_cache()
                time = []
                for k in range(num_frames):
                    self._conv_idx = [0]
                    tile = z[:, :, k : k + 1, i : i + tile_latent_min_height, j : j + tile_latent_min_width]
                    tile = self.post_quant_conv(tile)

                    tile = self._apply_token_dropout(tile)

                    decoded = self.decoder(
                        tile,
                        feat_cache=self._feat_map,
                        feat_idx=self._conv_idx,
                        reference_frame=reference_frame,
                        skip=skip,
                    )
                    time.append(decoded)
                row.append(torch.cat(time, dim=2))
            rows.append(row)
        self.clear_cache()

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        """
        Args:
            sample (`torch.Tensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample

        # Store reference frame if using reference attention
        if self.decoder.use_reference:
            idx = torch.randint(0, x.size(2), ()).item()
            self.reference_frame = x[:, :, idx : idx + 1, :, :].clone()
        else:
            self.reference_frame = None

        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior
