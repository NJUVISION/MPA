# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.
# Copyright (c) 2024, NJUVISION

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# * Neither the name of NJUVISION nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Any

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function

from timm.models.layers import DropPath, trunc_normal_

from .gdn import GDN
from natten import NeighborhoodAttention2D

__all__ = [
    "AttentionBlock",
    "MaskedConv2d",
    "CheckboardMaskedConv2d",
    "MultistageMaskedConv2d",
    "ResidualBlock",
    "ResidualBlockUpsample",
    "ResidualBlockWithStride",
    "conv3x3",
    "subpel_conv3x3",
    "QReLU",
    "ResViTBlock",
]


def gumbel_sigmoid(logits, training=True, tau=1.0, hard=False, threshold=0.5):
    if training:
        # ~Gumbel(0,1)`
        gumbels1 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels2 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        # Difference of two` gumbels because we apply a sigmoid
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
    else:
        y_soft = logits.sigmoid()

    if hard:
        # Straight through.
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).masked_fill_(y_soft > threshold, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


def batch_index_fill(x, x1, x2, idx1, idx2):
    B, N, C = x.size()
    B, N1, C = x1.size()
    B, N2, C = x2.size()

    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N
    idx2 = idx2 + offset * N

    x = x.reshape(B*N, C)

    x[idx1.reshape(-1)] = x1.reshape(B*N1, C)
    x[idx2.reshape(-1)] = x2.reshape(B*N2, C)

    x = x.reshape(B, N, C)
    return x


class MaskedConv2d(nn.Conv2d):
    r"""Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.

    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, h // 2 + 1 :] = 0

    def forward(self, x: Tensor) -> Tensor:
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)


class CheckboardMaskedConv2d(nn.Conv2d):
    """
    if kernel_size == (5, 5)
    then mask:
        [[0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.]]
    0: non-anchor
    1: anchor
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        self.mask[:, :, 0::2, 1::2] = 1
        self.mask[:, :, 1::2, 0::2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        out = super().forward(x)

        return out


class MultistageMaskedConv2d(nn.Conv2d):
    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        if mask_type == 'A': # 3x3
            self.mask[:, :, 0::2, 0::2] = 1
        elif mask_type == 'B':
            self.mask[:, :, 0::2, 1::2] = 1
            self.mask[:, :, 1::2, 0::2] = 1
        elif mask_type == 'C':
            self.mask[:, :, 0::2, 1::2] = 1
            self.mask[:, :, 1::2, :] = 1
        else:
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

    def forward(self, x: Tensor) -> Tensor:
        # TODO: weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r**2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = GDN(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return out


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        return out


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


class AttentionBlock(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out


class QReLU(Function):
    """QReLU

    Clamping input with given bit-depth range.
    Suppose that input data presents integer through an integer network
    otherwise any precision of input will simply clamp without rounding
    operation.

    Pre-computed scale with gamma function is used for backward computation.

    More details can be found in
    `"Integer networks for data compression with latent-variable models"
    <https://openreview.net/pdf?id=S1zz2i0cY7>`_,
    by Johannes Ballé, Nick Johnston and David Minnen, ICLR in 2019

    Args:
        input: a tensor data
        bit_depth: source bit-depth (used for clamping)
        beta: a parameter for modeling the gradient during backward computation
    """

    @staticmethod
    def forward(ctx, input, bit_depth, beta):
        # TODO(choih): allow to use adaptive scale instead of
        # pre-computed scale with gamma function
        ctx.alpha = 0.9943258522851727
        ctx.beta = beta
        ctx.max_value = 2**bit_depth - 1
        ctx.save_for_backward(input)

        return input.clamp(min=0, max=ctx.max_value)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        (input,) = ctx.saved_tensors

        grad_input = grad_output.clone()
        grad_sub = (
            torch.exp(
                (-ctx.alpha**ctx.beta)
                * torch.abs(2.0 * input / ctx.max_value - 1) ** ctx.beta
            )
            * grad_output.clone()
        )

        grad_input[input < 0] = grad_sub[input < 0]
        grad_input[input > ctx.max_value] = grad_sub[input > ctx.max_value]

        return grad_input, None, None


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FastMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
        )
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, task_idx=0):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SlowMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
        )
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, task_idx=0):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LinearProj(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, in_features),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, task_idx=0):
        x = self.fc1(x)
        x = self.drop(x)
        return x


class TaskMlp(nn.Module):
    def __init__(self, dim, mlp_ratio, act_layer, drop, mlp_cfg):
        super().__init__()
        self.mlp_list = []
        for mlp in mlp_cfg:
            if mlp is SlowMlp:
                self.mlp_list.append(mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop))
            elif mlp is FastMlp:
                self.mlp_list.append(mlp(in_features=dim, hidden_features=int(dim / mlp_ratio), act_layer=act_layer, drop=drop))
            elif mlp is LinearProj:
                self.mlp_list.append(mlp(in_features=dim, hidden_features=None, act_layer=act_layer, drop=drop))
            else:
                raise NotImplementedError
        self.mlp_list = nn.ModuleList(self.mlp_list)

    def forward(self, x, task_idx=0):
        x = self.mlp_list[task_idx](x)
        return x


class Predictor(nn.Module):
    """ Importance Score Predictor
    """
    def __init__(self, dim=384):
        super().__init__()
        self.levels = 8
        self.in_conv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 4),
            nn.GELU(),
        )
        self.out_conv = nn.Sequential(
            nn.Linear(dim // 4, dim // 16),
            nn.GELU(),
            nn.Linear(dim // 16, 1, bias=False),
        )
        self.log_base = 5
        self.shift = nn.Parameter(torch.zeros(self.levels), requires_grad=True)
        self.sf = 100.0    # scaling factor for fast convergence

    def gumbel_sigmoid(self, logits, tau=1.0, hard=False, threshold=0.5):
        if self.training:
            # ~Gumbel(0,1)`
            gumbels1 = (
                -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
                .exponential_()
                .log()
            )
            gumbels2 = (
                -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
                .exponential_()
                .log()
            )
            # Difference of two` gumbels because we apply a sigmoid
            gumbels1 = (logits + gumbels1 - gumbels2) / tau
            y_soft = gumbels1.sigmoid()
        else:
            y_soft = logits.sigmoid()

        if hard:
            # Straight through.
            y_hard = torch.zeros_like(
                logits, memory_format=torch.legacy_contiguous_format
            ).masked_fill_(y_soft > threshold, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret

    def forward(self, input_x, quality, q_task=None, mask=None):
        if self.training and mask is not None:
            x1, x2 = input_x
            input_x = x1 * mask + x2 * (1 - mask)
        else:
            x1 = input_x
            x2 = input_x

        x = self.in_conv(input_x)
        B, H, W, C = x.size()
        local_x = x[:, :, :, :C//2]
        global_x = torch.mean(x[:, :, :, C//2:], keepdim=True, dim=(1, 2))
        x = torch.cat([local_x, global_x.expand(B, H, W, C//2)], dim=-1)

        if self.training:
            if q_task is None:
                logits = self.out_conv(x) + self.shift[quality-1] * self.sf
            else:
                logits = self.out_conv(x) + self.shift[q_task-1] * self.sf
            mask = self.gumbel_sigmoid(logits, tau=1.0, hard=True, threshold=0.5)
            return [x1, x2], mask
        else:
            logits = self.out_conv(x)
            if q_task is None:
                ratio = (self.log_base**((quality - 1) / 7) - 1) / (self.log_base - 1)
            else:
                ratio = 1 - (q_task - 1) / 7
            score = logits.sigmoid().flatten(1)
            num_keep_node = int(score.shape[1] * ratio)
            idx = torch.argsort(score, dim=1, descending=True)
            idx1 = idx[:, :num_keep_node]
            idx2 = idx[:, num_keep_node:]
            return input_x, [idx1, idx2]


class NSABlock(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention2D(
            dim, kernel_size=kernel_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.levels = 8
        self.gamma_1 = nn.Parameter(torch.ones(size=[self.levels, dim]), requires_grad=True)
        self.gamma_2 = nn.Parameter(torch.ones(size=[self.levels, dim]), requires_grad=True)

    def forward(self, x, quality=None):
        if (not self.training) and (quality is not None):
            s = int(quality) - 1
            l = quality % 1
            # l = 0, Interpolated* = s+1; l = 1, Interpolated* = s
            if s == self.levels - 1:
                gamma_1 = torch.abs(self.gamma_1[s])
                gamma_2 = torch.abs(self.gamma_2[s])
            else:
                gamma_1 = torch.abs(self.gamma_1[s]).pow(1-l) * torch.abs(self.gamma_1[s+1]).pow(l)
                gamma_2 = torch.abs(self.gamma_2[s]).pow(1-l) * torch.abs(self.gamma_2[s+1]).pow(l)

        shortcut = x
        x = self.norm1(x)
        if quality is None:
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.training:
                x = torch.abs(self.gamma_1[quality-1]) * self.attn(x)
                x = shortcut + self.drop_path(x)
                x = x + self.drop_path(torch.abs(self.gamma_2[quality-1]) * self.mlp(self.norm2(x)))
            else:
                x = gamma_1 * self.attn(x)
                x = shortcut + self.drop_path(x)
                x = x + self.drop_path(gamma_2 * self.mlp(self.norm2(x)))
        return x


class AdaNSABlock(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_cfg=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention2D(
            dim, kernel_size=kernel_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if mlp_cfg is not None:
            self.fastmlp = TaskMlp(dim, mlp_ratio, act_layer, drop, mlp_cfg)
        else:
            self.fastmlp = FastMlp(in_features=dim, hidden_features=int(dim / mlp_ratio), act_layer=act_layer, drop=drop)
        self.levels = 8
        self.gamma_1 = nn.Parameter(torch.ones(size=[self.levels, dim]), requires_grad=True)
        self.gamma_2 = nn.Parameter(torch.ones(size=[self.levels, dim]), requires_grad=True)

    def forward(self, x, quality, mask=None, task_idx=0):
        if mask is not None and self.training:
            x1, x2 = x
            x = x1 * mask + x2 * (1 - mask)

        if not self.training:
            s = int(quality) - 1
            l = quality % 1
            # l = 0, Interpolated* = s+1; l = 1, Interpolated* = s
            if s == self.levels - 1:
                gamma_1 = torch.abs(self.gamma_1[s])
                gamma_2 = torch.abs(self.gamma_2[s])
            else:
                gamma_1 = torch.abs(self.gamma_1[s]).pow(1-l) * torch.abs(self.gamma_1[s+1]).pow(l)
                gamma_2 = torch.abs(self.gamma_2[s]).pow(1-l) * torch.abs(self.gamma_2[s+1]).pow(l)

        shortcut = x
        x = self.norm1(x)
        if self.training:
            x = torch.abs(self.gamma_1[quality-1]) * self.attn(x)
        else:
            x = gamma_1 * self.attn(x)
        x = shortcut + self.drop_path(x)

        if mask is None:
            if self.training:
                x = x + self.drop_path(torch.abs(self.gamma_2[quality-1]) * self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(gamma_2 * self.mlp(self.norm2(x)))
            return x
        else:
            if self.training:
                x1 = x * mask + x1 * (1 - mask)
                x2 = x * (1 - mask) + x2 * mask
                x1 = x + self.drop_path(torch.abs(self.gamma_2[quality-1]) * self.mlp(self.norm2(x1)))
                x2 = x + self.drop_path(torch.abs(self.gamma_2[quality-1]) * self.fastmlp(x2, task_idx=task_idx))
                return [x1, x2]
            else:
                B, H, W, C = x.shape
                x = x.flatten(1, 2)
                idx1, idx2 = mask

                x1 = batch_index_select(x, idx1)
                x2 = batch_index_select(x, idx2)
                x1 = self.drop_path(gamma_2 * self.mlp(self.norm2(x1)))
                x2 = self.drop_path(gamma_2 * self.fastmlp(x2, task_idx=task_idx))

                x0 = torch.zeros_like(x)
                x = x + batch_index_fill(x0, x1, x2, idx1, idx2)
                return x.reshape(B, H, W, C)


class BasicViTLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, kernel_size, mlp_ratio=4., 
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, mask_loc=None,
                 mlp_cfg=None):
        super().__init__()
        self.dim = dim
        self.depth = depth

        if mask_loc is None:
            self.blocks = nn.ModuleList([
                NSABlock(dim=dim,
                        num_heads=num_heads, kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                *[NSABlock(dim=dim,
                        num_heads=num_heads, kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer)
                for i in range(mask_loc[0])],
                *[AdaNSABlock(dim=dim,
                        num_heads=num_heads, kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer,
                        mlp_cfg=mlp_cfg)
                for i in range(mask_loc[0], depth)]])

        self.mask_loc = mask_loc
        if mask_loc is not None:
            if mlp_cfg is not None:
                self.num_predictors = len(mlp_cfg)
                self.score_predictor = []
                for _ in range(self.num_predictors):
                    predictor_list = [Predictor(dim) for i in range(len(mask_loc))]
                    self.score_predictor.append(nn.ModuleList(predictor_list))
                self.score_predictor = nn.ModuleList(self.score_predictor)
            else:
                self.num_predictors = None
                predictor_list = [Predictor(dim) for i in range(len(mask_loc))]
                self.score_predictor = nn.ModuleList(predictor_list)

    def forward(self, x, quality=None, q_task=None, task_idx=0):
        mask_loc_idx = 0
        mask = None
        decisions = []

        if self.mask_loc is None:
            for blk in self.blocks:
                x = blk(x, quality)
        else:
            for blk_idx, blk in enumerate(self.blocks):
                if blk_idx in self.mask_loc:
                    if self.num_predictors is not None:
                        x, mask = self.score_predictor[task_idx][mask_loc_idx](x, quality, q_task, mask)
                    else:
                        x, mask = self.score_predictor[mask_loc_idx](x, quality, q_task, mask)
                    mask_loc_idx += 1
                    decisions.append(mask)
                if blk_idx < self.mask_loc[0]:
                    x = blk(x, quality)
                else:
                    x = blk(x, quality, mask, task_idx)

            if isinstance(x, list):
                x = x[0] * mask + x[1] * (1 - mask)

        return x, decisions


class ResViTBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, kernel_size=7, mlp_ratio=4,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.2, norm_layer=nn.LayerNorm, mask_loc=None,
                 mlp_cfg=None):
        super(ResViTBlock, self).__init__()
        self.dim = dim

        self.residual_group = BasicViTLayer(dim=dim, depth=depth, num_heads=num_heads, kernel_size=kernel_size, mlp_ratio=mlp_ratio, 
                                            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                            drop_path=drop_path_rate, norm_layer=norm_layer, mask_loc=mask_loc,
                                            mlp_cfg=mlp_cfg)

    def forward(self, x, quality=None, q_task=None, task_idx=0):
        shortcut = x
        x, decisions = self.residual_group(x.permute(0, 2, 3, 1), quality, q_task=q_task, task_idx=task_idx)
        x = x.permute(0, 3, 1, 2) + shortcut
        return x, decisions
