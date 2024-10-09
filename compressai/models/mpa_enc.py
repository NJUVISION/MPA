# Copyright (c) 2024, NJUVISION

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
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

import math
import numpy as np

import torch
import torch.nn as nn

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers.layers import ResViTBlock, MultistageMaskedConv2d
from timm.models.layers import trunc_normal_

from .utils import conv, deconv, update_registered_buffers, quantize_ste, \
    Demultiplexer, Multiplexer, Demultiplexerv2, Multiplexerv2

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class MPAEnc(nn.Module):
    r"""
    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=128, M=320):
        super().__init__()

        depths = [2, 2, 6, 2, 2, 2]
        num_heads = [8, 12, 16, 20, 12, 12]
        kernel_size = 7
        mlp_ratio = 2.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        self.num_iters = 4
        self.gamma = self.gamma_func(mode="cosine")
        self.M = M
        self.levels = 8

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        self.g_a0 = conv(3, N, kernel_size=5, stride=2)
        self.g_a1 = ResViTBlock(dim=N,
                        depth=depths[0],
                        num_heads=num_heads[0],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:0]):sum(depths[:1])],
                        norm_layer=norm_layer,
                        mask_loc=[0, ],
        )
        self.g_a2 = conv(N, N*3//2, kernel_size=3, stride=2)
        self.g_a3 = ResViTBlock(dim=N*3//2,
                        depth=depths[1],
                        num_heads=num_heads[1],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:1]):sum(depths[:2])],
                        norm_layer=norm_layer,
                        mask_loc=[0, ],
        )
        self.g_a4 = conv(N*3//2, N*2, kernel_size=3, stride=2)
        self.g_a5 = ResViTBlock(dim=N*2,
                        depth=depths[2],
                        num_heads=num_heads[2],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        mask_loc=[0, ],
        )
        self.g_a6 = conv(N*2, M, kernel_size=3, stride=2)
        self.g_a7 = ResViTBlock(dim=M,
                        depth=depths[3],
                        num_heads=num_heads[3],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
        )

        self.h_a0 = conv(M, N*3//2, kernel_size=3, stride=2)
        self.h_a1 = ResViTBlock(dim=N*3//2,
                        depth=depths[4],
                        num_heads=num_heads[4],
                        kernel_size=kernel_size//2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:4]):sum(depths[:5])],
                        norm_layer=norm_layer,
        )
        self.h_a2 = conv(N*3//2, N*3//2, kernel_size=3, stride=2)
        self.h_a3 = ResViTBlock(dim=N*3//2,
                        depth=depths[5],
                        num_heads=num_heads[5],
                        kernel_size=kernel_size//2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:5]):sum(depths[:6])],
                        norm_layer=norm_layer,
        )

        depths = depths[::-1]
        num_heads = num_heads[::-1]
        self.h_s0 = ResViTBlock(dim=N*3//2,
                        depth=depths[0],
                        num_heads=num_heads[0],
                        kernel_size=kernel_size//2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:0]):sum(depths[:1])],
                        norm_layer=norm_layer,
        )
        self.h_s1 = deconv(N*3//2, N*3//2, kernel_size=3, stride=2)
        self.h_s2 = ResViTBlock(dim=N*3//2,
                        depth=depths[1],
                        num_heads=num_heads[1],
                        kernel_size=kernel_size//2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:1]):sum(depths[:2])],
                        norm_layer=norm_layer,
        )
        self.h_s3 = deconv(N*3//2, M*2, kernel_size=3, stride=2)
        
        self.g_s0 = ResViTBlock(dim=M,
                        depth=depths[2],
                        num_heads=num_heads[2],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
        )
        self.g_s1 = deconv(M, N*2, kernel_size=3, stride=2)
        self.g_s2 = ResViTBlock(dim=N*2,
                        depth=depths[3],
                        num_heads=num_heads[3],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
        )
        self.g_s3 = deconv(N*2, N*3//2, kernel_size=3, stride=2)
        self.g_s4 = ResViTBlock(dim=N*3//2,
                        depth=depths[4],
                        num_heads=num_heads[4],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:4]):sum(depths[:5])],
                        norm_layer=norm_layer,
        )
        self.g_s5 = deconv(N*3//2, N, kernel_size=3, stride=2)
        self.g_s6 = ResViTBlock(dim=N,
                        depth=depths[5],
                        num_heads=num_heads[5],
                        kernel_size=kernel_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:5]):sum(depths[:6])],
                        norm_layer=norm_layer,
        )
        self.g_s7 = deconv(N, 3, kernel_size=5, stride=2)

        self.entropy_bottleneck = EntropyBottleneck(N*3//2)
        self.gaussian_conditional = GaussianConditional(None)

        self.sc_transform_1 = MultistageMaskedConv2d(
                int(M - np.ceil(self.gamma(1/self.num_iters)*M)), int(M - np.ceil(self.gamma(1/self.num_iters)*M))*2, 
                kernel_size=3, padding=1, stride=1, mask_type='A'
        )
        self.sc_transform_2 = MultistageMaskedConv2d(
                int(M - np.ceil(self.gamma(1/self.num_iters)*M)), int(M - np.ceil(self.gamma(1/self.num_iters)*M))*2, 
                kernel_size=5, padding=2, stride=1, mask_type='B'
        )
        self.sc_transform_3 = MultistageMaskedConv2d(
                int(M - np.ceil(self.gamma(1/self.num_iters)*M)), int(M - np.ceil(self.gamma(1/self.num_iters)*M))*2, 
                kernel_size=5, padding=2, stride=1, mask_type='C'
        )
        self.sc_transform_4 = MultistageMaskedConv2d(
                int(np.ceil(self.gamma(1/self.num_iters)*M)-np.ceil(self.gamma(2/self.num_iters)*M)), 
                int(np.ceil(self.gamma(1/self.num_iters)*M)-np.ceil(self.gamma(2/self.num_iters)*M))*2,
                kernel_size=5, padding=2, stride=1, mask_type='B'
        )
        self.sc_transform_5 = MultistageMaskedConv2d(
                int(np.ceil(self.gamma(2/self.num_iters)*M)-np.ceil(self.gamma(3/self.num_iters)*M)), 
                int(np.ceil(self.gamma(2/self.num_iters)*M)-np.ceil(self.gamma(3/self.num_iters)*M))*2,
                kernel_size=5, padding=2, stride=1, mask_type='B'
        )

        self.entropy_parameters_1 = nn.Sequential(
                conv(M*2 + int(M - np.ceil(self.gamma(1/self.num_iters)*M))*12//3, 
                int(M - np.ceil(self.gamma(1/self.num_iters)*M))*10//3, 1, 1),
                nn.GELU(),
                conv(int(M - np.ceil(self.gamma(1/self.num_iters)*M))*10//3, 
                int(M - np.ceil(self.gamma(1/self.num_iters)*M))*8//3, 1, 1),
                nn.GELU(),
                conv(int(M - np.ceil(self.gamma(1/self.num_iters)*M))*8//3, 
                int(M - np.ceil(self.gamma(1/self.num_iters)*M))*6//3, 1, 1),
        )
        self.entropy_parameters_2 = nn.Sequential(
                conv(M*2 + int(np.ceil(self.gamma(1/self.num_iters)*M) - np.ceil(self.gamma(2/self.num_iters)*M))*12//3, 
                int(np.ceil(self.gamma(1/self.num_iters)*M) - np.ceil(self.gamma(2/self.num_iters)*M))*10//3, 1, 1),
                nn.GELU(),
                conv(int(np.ceil(self.gamma(1/self.num_iters)*M) - np.ceil(self.gamma(2/self.num_iters)*M))*10//3, 
                int(np.ceil(self.gamma(1/self.num_iters)*M) - np.ceil(self.gamma(2/self.num_iters)*M))*8//3, 1, 1),
                nn.GELU(),
                conv(int(np.ceil(self.gamma(1/self.num_iters)*M) - np.ceil(self.gamma(2/self.num_iters)*M))*8//3, 
                int(np.ceil(self.gamma(1/self.num_iters)*M) - np.ceil(self.gamma(2/self.num_iters)*M))*6//3, 1, 1),
        ) 
        self.entropy_parameters_3 = nn.Sequential(
                conv(M*2 + int(np.ceil(self.gamma(2/self.num_iters)*M) - np.ceil(self.gamma(3/self.num_iters)*M))*12//3, 
                int(np.ceil(self.gamma(2/self.num_iters)*M) - np.ceil(self.gamma(3/self.num_iters)*M))*10//3, 1, 1),
                nn.GELU(),
                conv(int(np.ceil(self.gamma(2/self.num_iters)*M) - np.ceil(self.gamma(3/self.num_iters)*M))*10//3, 
                int(np.ceil(self.gamma(2/self.num_iters)*M) - np.ceil(self.gamma(3/self.num_iters)*M))*8//3, 1, 1),
                nn.GELU(),
                conv(int(np.ceil(self.gamma(2/self.num_iters)*M) - np.ceil(self.gamma(3/self.num_iters)*M))*8//3, 
                int(np.ceil(self.gamma(2/self.num_iters)*M) - np.ceil(self.gamma(3/self.num_iters)*M))*6//3, 1, 1),
        ) 
        self.entropy_parameters_4 = nn.Sequential(
                conv(M*2 + int(M-(np.ceil(self.gamma(0)*M)-np.ceil(self.gamma(3/self.num_iters)*M)))*6//3, 
                int(M-(np.ceil(self.gamma(0)*M)-np.ceil(self.gamma(3/self.num_iters)*M)))*6//3, 1, 1),
                nn.GELU(),
                conv(int(M-(np.ceil(self.gamma(0)*M)-np.ceil(self.gamma(3/self.num_iters)*M)))*6//3, 
                int(M-(np.ceil(self.gamma(0)*M)-np.ceil(self.gamma(3/self.num_iters)*M)))*6//3, 1, 1),
                nn.GELU(),
                conv(int(M-(np.ceil(self.gamma(0)*M)-np.ceil(self.gamma(3/self.num_iters)*M)))*6//3, 
                int(M-(np.ceil(self.gamma(0)*M)-np.ceil(self.gamma(3/self.num_iters)*M)))*6//3, 1, 1),
        ) 
        
        self.cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M*2 + int(M - np.ceil(self.gamma(i/self.num_iters)*M)), 
                int(np.ceil(self.gamma(i/self.num_iters)*M)-np.ceil(self.gamma((i+1)/self.num_iters)*M))
                if i != self.num_iters-1 else int(M-(np.ceil(self.gamma(0)*M)-np.ceil(self.gamma(i/self.num_iters)*M))),
                stride=1, kernel_size=5),
                nn.GELU(),
                conv(int(np.ceil(self.gamma(i/self.num_iters)*M)-np.ceil(self.gamma((i+1)/self.num_iters)*M))
                if i != self.num_iters-1 else int(M-(np.ceil(self.gamma(0)*M)-np.ceil(self.gamma(i/self.num_iters)*M))), 
                int(np.ceil(self.gamma(i/self.num_iters)*M)-np.ceil(self.gamma((i+1)/self.num_iters)*M))
                if i != self.num_iters-1 else int(M-(np.ceil(self.gamma(0)*M)-np.ceil(self.gamma(i/self.num_iters)*M))), 
                stride=1, kernel_size=5),
                nn.GELU(),
                conv(int(np.ceil(self.gamma(i/self.num_iters)*M)-np.ceil(self.gamma((i+1)/self.num_iters)*M))
                if i != self.num_iters-1 else int(M-(np.ceil(self.gamma(0)*M)-np.ceil(self.gamma(i/self.num_iters)*M))), 
                int(np.ceil(self.gamma(i/self.num_iters)*M)-np.ceil(self.gamma((i+1)/self.num_iters)*M))*2
                if i != self.num_iters-1 else int(M-(np.ceil(self.gamma(0)*M)-np.ceil(self.gamma(i/self.num_iters)*M)))*2, 
                stride=1, kernel_size=3),
            ) for i in range(self.num_iters)
        )

        self.Gain = nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
        self.InverseGain = nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
        self.HyperGain = nn.Parameter(torch.ones(size=[self.levels, N*3//2]), requires_grad=True)
        self.InverseHyperGain = nn.Parameter(torch.ones(size=[self.levels, N*3//2]), requires_grad=True)

        self.apply(self._init_weights)

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r**2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError  

    def g_a(self, x, quality):
        x = self.g_a0(x)
        x, decisions_g_a1 = self.g_a1(x, quality=quality)
        x = self.g_a2(x)
        x, decisions_g_a3 = self.g_a3(x, quality=quality)
        x = self.g_a4(x)
        x, decisions_g_a5 = self.g_a5(x, quality=quality)
        x = self.g_a6(x)
        x, decisions_g_a7 = self.g_a7(x, quality=quality)
        return x, [*decisions_g_a1, *decisions_g_a3, *decisions_g_a5, *decisions_g_a7]

    def g_s(self, x, quality, q_task=None):
        x, decisions_g_s0 = self.g_s0(x, quality=quality, q_task=q_task)
        x = self.g_s1(x)
        x, decisions_g_s2 = self.g_s2(x, quality=quality, q_task=q_task)
        x = self.g_s3(x)
        x, decisions_g_s4 = self.g_s4(x, quality=quality, q_task=q_task)
        x = self.g_s5(x)
        x, decisions_g_s6 = self.g_s6(x, quality=quality, q_task=q_task)
        x = self.g_s7(x)
        return x, [*decisions_g_s6, *decisions_g_s4, *decisions_g_s2, *decisions_g_s0]

    def h_a(self, x, quality):
        x = self.h_a0(x)
        x, decisions_h_a1 = self.h_a1(x, quality=quality)
        x = self.h_a2(x)
        x, decisions_h_a3 = self.h_a3(x, quality=quality)
        return x, [*decisions_h_a1, *decisions_h_a3]

    def h_s(self, x, quality):
        x, decisions_h_s0 = self.h_s0(x, quality=quality)
        x = self.h_s1(x)
        x, decisions_h_s2 = self.h_s2(x, quality=quality)
        x = self.h_s3(x)
        return x, [*decisions_h_s2, *decisions_h_s0]

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, quality=1, q_task=1):
        y, enc_decisions = self.g_a(x, quality)
        y = y * torch.abs(self.Gain[quality-1]).unsqueeze(0).unsqueeze(2).unsqueeze(3) # Gain[s]: [M]  -->  [1,M,1,1]
        z, henc_decisions = self.h_a(y, quality)
        z = z * torch.abs(self.HyperGain[quality-1]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset

        z_hat = z_hat * torch.abs(self.InverseHyperGain[quality-1]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        params, hdec_decisions = self.h_s(z_hat, quality)
        
        slice_list = []
        N = y.shape[1]
        for t in range(1, self.num_iters+1):
            if t == self.num_iters:
                n = 0
            else:
                n = np.ceil(self.gamma(t/(self.num_iters)) * y.shape[1])

            slice_list.append(int(N-n))
            N = n

        y_slices = y.split(tuple(slice_list), 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):

            if slice_index == 0:
                support_slices = torch.cat([params] + y_hat_slices, dim=1)
                cc_params = self.cc_transforms[slice_index](support_slices)

                sc_params_1 = torch.zeros_like(cc_params)
                sc_params = sc_params_1
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                y_0 = y_hat_slice.clone()
                y_0[:, :, 0::2, 1::2] = 0
                y_0[:, :, 1::2, :] = 0
                sc_params_2 = self.sc_transform_1(y_0)
                sc_params_2[:, :, 0::2, :] = 0
                sc_params_2[:, :, 1::2, 0::2] = 0
                sc_params = sc_params_1 + sc_params_2
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                y_1 = y_hat_slice.clone()
                y_1[:, :, 0::2, 1::2] = 0
                y_1[:, :, 1::2, 0::2] = 0
                sc_params_3 = self.sc_transform_2(y_1)
                sc_params_3[:, :, 0::2, 0::2] = 0
                sc_params_3[:, :, 1::2, :] = 0
                sc_params = sc_params_1 + sc_params_2 + sc_params_3
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                y_2 = y_hat_slice.clone()
                y_2[:, :, 1::2, 0::2] = 0
                sc_params_4 = self.sc_transform_3(y_2)
                sc_params_4[:, :, 0::2, :] = 0
                sc_params_4[:, :, 1::2, 1::2] = 0
                sc_params = sc_params_1 + sc_params_2 + sc_params_3 + sc_params_4
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat
                y_hat_slices.append(y_hat_slice)

                _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat, means=means_hat)
                y_likelihood.append(y_slice_likelihood)

            elif slice_index == 1:
                support_slices = torch.cat([params] + y_hat_slices, dim=1)
                cc_params = self.cc_transforms[slice_index](support_slices)

                sc_params = torch.zeros_like(cc_params)
                gaussian_params = self.entropy_parameters_2(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                y_half = y_hat_slice.clone()
                y_half[:, :, 0::2, 0::2] = 0
                y_half[:, :, 1::2, 1::2] = 0

                sc_params = self.sc_transform_4(y_half)
                sc_params[:, :, 0::2, 1::2] = 0
                sc_params[:, :, 1::2, 0::2] = 0
                
                gaussian_params = self.entropy_parameters_2(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat
                y_hat_slices.append(y_hat_slice)

                _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat, means=means_hat)
                y_likelihood.append(y_slice_likelihood)

            elif slice_index == 2:
                support_slices = torch.cat([params] + y_hat_slices, dim=1)
                cc_params = self.cc_transforms[slice_index](support_slices)

                sc_params = torch.zeros_like(cc_params)
                gaussian_params = self.entropy_parameters_3(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat

                y_half = y_hat_slice.clone()
                y_half[:, :, 0::2, 1::2] = 0
                y_half[:, :, 1::2, 0::2] = 0

                sc_params = self.sc_transform_5(y_half)
                sc_params[:, :, 0::2, 0::2] = 0
                sc_params[:, :, 1::2, 1::2] = 0

                gaussian_params = self.entropy_parameters_3(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat
                y_hat_slices.append(y_hat_slice)

                _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat, means=means_hat)
                y_likelihood.append(y_slice_likelihood)

            else:
                support_slices = torch.cat([params] + y_hat_slices, dim=1)
                cc_params = self.cc_transforms[slice_index](support_slices)

                gaussian_params = self.entropy_parameters_4(
                    torch.cat((params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_hat_slice = quantize_ste(y_slice - means_hat) + means_hat
                y_hat_slices.append(y_hat_slice)

                _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat, means=means_hat)
                y_likelihood.append(y_slice_likelihood)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        # Generate the image reconstruction.
        # y_hat = y_hat * torch.abs(self.InverseGain[quality-1]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat, dec_decisions = self.g_s(y_hat * torch.abs(self.InverseGain[quality-1]).unsqueeze(0).unsqueeze(2).unsqueeze(3), quality, q_task)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "decisions": {"enc": enc_decisions, "dec": dec_decisions, "henc": henc_decisions, "hdec": hdec_decisions},
            "latent": {"y": y, "y_hat": y_hat},
        }

    def forward(self, x, quality=1, q_task=1, task_idx=0):
        x = self.forward_features(x, quality, q_task)
        return x

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a0.weight"].size(0)
        M = state_dict["g_a6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x, quality=1):
        assert quality >= 1 and quality <= self.levels, f"quality should within range [1, {self.levels}], but get quality={quality}"
        s = int(quality) - 1
        l = quality % 1
        # l = 0, Interpolated* = s+1; l = 1, Interpolated* = s
        if s == self.levels - 1:
            InterpolatedGain = torch.abs(self.Gain[s])
            InterpolatedHyperGain = torch.abs(self.HyperGain[s])
            InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s])
        else:
            InterpolatedGain = torch.abs(self.Gain[s]).pow(1-l) * torch.abs(self.Gain[s+1]).pow(l)
            InterpolatedHyperGain = torch.abs(self.HyperGain[s]).pow(1-l) * torch.abs(self.HyperGain[s+1]).pow(l)
            InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)

        y, _ = self.g_a(x, quality)
        y = y * InterpolatedGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        z, _ = self.h_a(y, quality)
        z = z * InterpolatedHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        params, _ = self.h_s(z_hat, quality)

        slice_list = []
        N = y.shape[1]
        for t in range(1, self.num_iters+1):
            if t == self.num_iters:
                n = 0
            else:
                n = np.ceil(self.gamma(t/(self.num_iters)) * y.shape[1])

            slice_list.append(int(N-n))
            N = n

        y_slices = y.split(tuple(slice_list), 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            
            if slice_index == 0:
                cc_params = self.cc_transforms[slice_index](params)

                y_slice_0, y_slice_1, y_slice_2, y_slice_3 = Demultiplexerv2(y_slice)
                
                sc_params_1 = torch.zeros(y_slice.shape[0], y_slice.shape[1]*2, y_slice.shape[2], y_slice.shape[3]).to(z_hat.device)
                sc_params = sc_params_1
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                scales_hat_0, _, _, _ = Demultiplexerv2(scales_hat)
                means_hat_0, _, _, _ = Demultiplexerv2(means_hat)
                index_0 = self.gaussian_conditional.build_indexes(scales_hat_0)
                y_q_slice_0 = self.gaussian_conditional.quantize(y_slice_0, "symbols", means_hat_0)
                y_hat_slice_0 = y_q_slice_0 + means_hat_0

                symbols_list.extend(y_q_slice_0.reshape(-1).tolist())
                indexes_list.extend(index_0.reshape(-1).tolist())

                y_hat_slice = Multiplexerv2(y_hat_slice_0, torch.zeros_like(y_hat_slice_0), 
                                            torch.zeros_like(y_hat_slice_0), torch.zeros_like(y_hat_slice_0))
                sc_params_2 = self.sc_transform_1(y_hat_slice)
                sc_params_2[:, :, 0::2, :] = 0
                sc_params_2[:, :, 1::2, 0::2] = 0
                sc_params = sc_params_1 + sc_params_2
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, scales_hat_1, _, _ = Demultiplexerv2(scales_hat)
                _, means_hat_1, _, _ = Demultiplexerv2(means_hat)
                index_1 = self.gaussian_conditional.build_indexes(scales_hat_1)
                y_q_slice_1 = self.gaussian_conditional.quantize(y_slice_1, "symbols", means_hat_1)
                y_hat_slice_1 = y_q_slice_1 + means_hat_1

                symbols_list.extend(y_q_slice_1.reshape(-1).tolist())
                indexes_list.extend(index_1.reshape(-1).tolist())

                y_hat_slice = Multiplexerv2(y_hat_slice_0, y_hat_slice_1, 
                                            torch.zeros_like(y_hat_slice_0), torch.zeros_like(y_hat_slice_0))
                sc_params_3 = self.sc_transform_2(y_hat_slice)
                sc_params_3[:, :, 0::2, 0::2] = 0
                sc_params_3[:, :, 1::2, :] = 0
                sc_params = sc_params_1 + sc_params_2 + sc_params_3
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, _, scales_hat_2, _ = Demultiplexerv2(scales_hat)
                _, _, means_hat_2, _ = Demultiplexerv2(means_hat)
                index_2 = self.gaussian_conditional.build_indexes(scales_hat_2)
                y_q_slice_2 = self.gaussian_conditional.quantize(y_slice_2, "symbols", means_hat_2)
                y_hat_slice_2 = y_q_slice_2 + means_hat_2

                symbols_list.extend(y_q_slice_2.reshape(-1).tolist())
                indexes_list.extend(index_2.reshape(-1).tolist())

                y_hat_slice = Multiplexerv2(y_hat_slice_0, y_hat_slice_1, 
                                            y_hat_slice_2, torch.zeros_like(y_hat_slice_0))
                sc_params_4 = self.sc_transform_3(y_hat_slice)
                sc_params_4[:, :, 0::2, :] = 0
                sc_params_4[:, :, 1::2, 1::2] = 0
                sc_params = sc_params_1 + sc_params_2 + sc_params_3 + sc_params_4
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, _, _, scales_hat_3 = Demultiplexerv2(scales_hat)
                _, _, _, means_hat_3 = Demultiplexerv2(means_hat)
                index_3 = self.gaussian_conditional.build_indexes(scales_hat_3)
                y_q_slice_3 = self.gaussian_conditional.quantize(y_slice_3, "symbols", means_hat_3)
                y_hat_slice_3 = y_q_slice_3 + means_hat_3

                symbols_list.extend(y_q_slice_3.reshape(-1).tolist())
                indexes_list.extend(index_3.reshape(-1).tolist())

                y_hat_slice = Multiplexerv2(y_hat_slice_0, y_hat_slice_1, y_hat_slice_2, y_hat_slice_3)
                y_hat_slices.append(y_hat_slice)

            elif slice_index == 1:
                support_slices = torch.cat([params] + [y_hat_slices[i] for i in range(slice_index)], dim=1)
                cc_params = self.cc_transforms[slice_index](support_slices)

                y_slice_anchor, y_slice_non_anchor = Demultiplexer(y_slice)

                zero_sc_params = torch.zeros(y_slice.shape[0], y_slice.shape[1]*2, y_slice.shape[2], y_slice.shape[3]).to(z_hat.device)
                gaussian_params = self.entropy_parameters_2(
                    torch.cat((params, zero_sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                scales_hat_anchor, _ = Demultiplexer(scales_hat)
                means_hat_anchor, _ = Demultiplexer(means_hat)
                index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
                y_q_slice_anchor = self.gaussian_conditional.quantize(y_slice_anchor, "symbols", means_hat_anchor)
                y_hat_slice_anchor = y_q_slice_anchor + means_hat_anchor

                symbols_list.extend(y_q_slice_anchor.reshape(-1).tolist())
                indexes_list.extend(index_anchor.reshape(-1).tolist())

                y_hat_slice = Multiplexer(y_hat_slice_anchor, torch.zeros_like(y_hat_slice_anchor))
                sc_params = self.sc_transform_4(y_hat_slice)
                sc_params[:, :, 0::2, 1::2] = 0
                sc_params[:, :, 1::2, 0::2] = 0
                
                gaussian_params = self.entropy_parameters_2(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, scales_hat_non_anchor = Demultiplexer(scales_hat)
                _, means_hat_non_anchor = Demultiplexer(means_hat)
                index_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
                y_q_slice_non_anchor = self.gaussian_conditional.quantize(y_slice_non_anchor, "symbols", means_hat_non_anchor)
                y_hat_slice_non_anchor = y_q_slice_non_anchor + means_hat_non_anchor

                symbols_list.extend(y_q_slice_non_anchor.reshape(-1).tolist())
                indexes_list.extend(index_non_anchor.reshape(-1).tolist())

                y_hat_slice = Multiplexer(y_hat_slice_anchor, y_hat_slice_non_anchor)
                y_hat_slices.append(y_hat_slice)

            elif slice_index == 2:
                support_slices = torch.cat([params] + [y_hat_slices[i] for i in range(slice_index)], dim=1)
                cc_params = self.cc_transforms[slice_index](support_slices)

                y_slice_non_anchor, y_slice_anchor = Demultiplexer(y_slice)

                zero_sc_params = torch.zeros(y_slice.shape[0], y_slice.shape[1]*2, y_slice.shape[2], y_slice.shape[3]).to(z_hat.device)
                gaussian_params = self.entropy_parameters_3(
                    torch.cat((params, zero_sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, scales_hat_anchor = Demultiplexer(scales_hat)
                _, means_hat_anchor = Demultiplexer(means_hat)
                index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
                y_q_slice_anchor = self.gaussian_conditional.quantize(y_slice_anchor, "symbols", means_hat_anchor)
                y_hat_slice_anchor = y_q_slice_anchor + means_hat_anchor

                symbols_list.extend(y_q_slice_anchor.reshape(-1).tolist())
                indexes_list.extend(index_anchor.reshape(-1).tolist())

                y_hat_slice = Multiplexer(torch.zeros_like(y_hat_slice_anchor), y_hat_slice_anchor)
                sc_params = self.sc_transform_5(y_hat_slice)
                sc_params[:, :, 0::2, 0::2] = 0
                sc_params[:, :, 1::2, 1::2] = 0

                gaussian_params = self.entropy_parameters_3(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                scales_hat_non_anchor, _ = Demultiplexer(scales_hat)
                means_hat_non_anchor, _ = Demultiplexer(means_hat)
                index_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
                y_q_slice_non_anchor = self.gaussian_conditional.quantize(y_slice_non_anchor, "symbols", means_hat_non_anchor)
                y_hat_slice_non_anchor = y_q_slice_non_anchor + means_hat_non_anchor

                symbols_list.extend(y_q_slice_non_anchor.reshape(-1).tolist())
                indexes_list.extend(index_non_anchor.reshape(-1).tolist())

                y_hat_slice = Multiplexer(y_hat_slice_non_anchor, y_hat_slice_anchor)
                y_hat_slices.append(y_hat_slice)

            else:
                support_slices = torch.cat([params] + [y_hat_slices[i] for i in range(slice_index)], dim=1)
                cc_params = self.cc_transforms[slice_index](support_slices)

                gaussian_params = self.entropy_parameters_4(
                    torch.cat((params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                index = self.gaussian_conditional.build_indexes(scales_hat)
                y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", means_hat)
                y_hat_slice = y_q_slice + means_hat

                symbols_list.extend(y_q_slice.reshape(-1).tolist())
                indexes_list.extend(index.reshape(-1).tolist())

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], 
                "shape": z.size()[-2:]
                }

    def decompress(self, strings, shape, quality=1, q_task=1, task_idx=0):
        assert isinstance(strings, list) and len(strings) == 2
        assert quality >= 1 and quality <= self.levels, f"quality should within range [1, {self.levels}], but get quality={quality}"
        s = int(quality) - 1
        l = quality % 1
        # l = 0, Interpolated* = s+1; l = 1, Interpolated* = s
        if s == self.levels - 1:
            InterpolatedInverseGain = torch.abs(self.InverseGain[s])
            InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s])
        else:
            InterpolatedInverseGain = torch.abs(self.InverseGain[s]).pow(1-l) * torch.abs(self.InverseGain[s+1]).pow(l)
            InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        params, _ = self.h_s(z_hat, quality)

        slice_list = []
        N = self.M
        for t in range(1, self.num_iters+1):
            if t == self.num_iters:
                n = 0
            else:
                n = np.ceil(self.gamma(t/(self.num_iters)) * self.M)

            slice_list.append(int(N-n))
            N = n

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_iters):

            if slice_index == 0:
                cc_params = self.cc_transforms[slice_index](params)

                sc_params_1 = torch.zeros(z_hat.shape[0], slice_list[slice_index]*2, z_hat.shape[2]*4, z_hat.shape[3]*4).to(z_hat.device)
                sc_params = sc_params_1
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                scales_hat_0, _, _, _ = Demultiplexerv2(scales_hat)
                means_hat_0, _, _, _ = Demultiplexerv2(means_hat)
                index_0 = self.gaussian_conditional.build_indexes(scales_hat_0)
                rv = decoder.decode_stream(index_0.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(z_hat.shape[0], -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
                y_hat_slice_0 = self.gaussian_conditional.dequantize(rv, means_hat_0)

                y_hat_slice = Multiplexerv2(y_hat_slice_0, torch.zeros_like(y_hat_slice_0), 
                                            torch.zeros_like(y_hat_slice_0), torch.zeros_like(y_hat_slice_0))
                sc_params_2 = self.sc_transform_1(y_hat_slice)
                sc_params_2[:, :, 0::2, :] = 0
                sc_params_2[:, :, 1::2, 0::2] = 0
                sc_params = sc_params_1 + sc_params_2
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, scales_hat_1, _, _ = Demultiplexerv2(scales_hat)
                _, means_hat_1, _, _ = Demultiplexerv2(means_hat)
                index_1 = self.gaussian_conditional.build_indexes(scales_hat_1)
                rv = decoder.decode_stream(index_1.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(z_hat.shape[0], -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
                y_hat_slice_1 = self.gaussian_conditional.dequantize(rv, means_hat_1)

                y_hat_slice = Multiplexerv2(y_hat_slice_0, y_hat_slice_1, 
                                            torch.zeros_like(y_hat_slice_0), torch.zeros_like(y_hat_slice_0))
                sc_params_3 = self.sc_transform_2(y_hat_slice)
                sc_params_3[:, :, 0::2, 0::2] = 0
                sc_params_3[:, :, 1::2, :] = 0
                sc_params = sc_params_1 + sc_params_2 + sc_params_3
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, _, scales_hat_2, _ = Demultiplexerv2(scales_hat)
                _, _, means_hat_2, _ = Demultiplexerv2(means_hat)
                index_2 = self.gaussian_conditional.build_indexes(scales_hat_2)
                rv = decoder.decode_stream(index_2.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(z_hat.shape[0], -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
                y_hat_slice_2 = self.gaussian_conditional.dequantize(rv, means_hat_2)

                y_hat_slice = Multiplexerv2(y_hat_slice_0, y_hat_slice_1, 
                                            y_hat_slice_2, torch.zeros_like(y_hat_slice_0))
                sc_params_4 = self.sc_transform_3(y_hat_slice)
                sc_params_4[:, :, 0::2, :] = 0
                sc_params_4[:, :, 1::2, 1::2] = 0
                sc_params = sc_params_1 + sc_params_2 + sc_params_3 + sc_params_4
                gaussian_params = self.entropy_parameters_1(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, _, _, scales_hat_3 = Demultiplexerv2(scales_hat)
                _, _, _, means_hat_3 = Demultiplexerv2(means_hat)
                index_3 = self.gaussian_conditional.build_indexes(scales_hat_3)
                rv = decoder.decode_stream(index_3.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(z_hat.shape[0], -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
                y_hat_slice_3 = self.gaussian_conditional.dequantize(rv, means_hat_3)

                y_hat_slice = Multiplexerv2(y_hat_slice_0, y_hat_slice_1, y_hat_slice_2, y_hat_slice_3)
                y_hat_slices.append(y_hat_slice)

            elif slice_index == 1:
                support_slices = torch.cat([params] + [y_hat_slices[i] for i in range(slice_index)], dim=1)
                cc_params = self.cc_transforms[slice_index](support_slices)

                zero_sc_params = torch.zeros(z_hat.shape[0], slice_list[slice_index]*2, z_hat.shape[2]*4, z_hat.shape[3]*4).to(z_hat.device)
                gaussian_params = self.entropy_parameters_2(
                    torch.cat((params, zero_sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                scales_hat_anchor, _ = Demultiplexer(scales_hat)
                means_hat_anchor, _ = Demultiplexer(means_hat)
                index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
                rv = decoder.decode_stream(index_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(z_hat.shape[0], -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
                y_hat_slice_anchor = self.gaussian_conditional.dequantize(rv, means_hat_anchor)

                y_hat_slice = Multiplexer(y_hat_slice_anchor, torch.zeros_like(y_hat_slice_anchor))
                sc_params = self.sc_transform_4(y_hat_slice)
                sc_params[:, :, 0::2, 1::2] = 0
                sc_params[:, :, 1::2, 0::2] = 0
                
                gaussian_params = self.entropy_parameters_2(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                
                _, scales_hat_non_anchor = Demultiplexer(scales_hat)
                _, means_hat_non_anchor = Demultiplexer(means_hat)
                index_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
                rv = decoder.decode_stream(index_non_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(z_hat.shape[0], -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
                y_hat_slice_non_anchor = self.gaussian_conditional.dequantize(rv, means_hat_non_anchor)
                
                y_hat_slice = Multiplexer(y_hat_slice_anchor, y_hat_slice_non_anchor)
                y_hat_slices.append(y_hat_slice)

            elif slice_index == 2:
                support_slices = torch.cat([params] + [y_hat_slices[i] for i in range(slice_index)], dim=1)
                cc_params = self.cc_transforms[slice_index](support_slices)

                zero_sc_params = torch.zeros(z_hat.shape[0], slice_list[slice_index]*2, z_hat.shape[2]*4, z_hat.shape[3]*4).to(z_hat.device)
                gaussian_params = self.entropy_parameters_3(
                    torch.cat((params, zero_sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                _, scales_hat_anchor = Demultiplexer(scales_hat)
                _, means_hat_anchor = Demultiplexer(means_hat)
                index_anchor = self.gaussian_conditional.build_indexes(scales_hat_anchor)
                rv = decoder.decode_stream(index_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(z_hat.shape[0], -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
                y_hat_slice_anchor = self.gaussian_conditional.dequantize(rv, means_hat_anchor)

                y_hat_slice = Multiplexer(torch.zeros_like(y_hat_slice_anchor), y_hat_slice_anchor)
                sc_params = self.sc_transform_5(y_hat_slice)
                sc_params[:, :, 0::2, 0::2] = 0
                sc_params[:, :, 1::2, 1::2] = 0
                
                gaussian_params = self.entropy_parameters_3(
                    torch.cat((params, sc_params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                
                scales_hat_non_anchor, _ = Demultiplexer(scales_hat)
                means_hat_non_anchor, _ = Demultiplexer(means_hat)
                index_non_anchor = self.gaussian_conditional.build_indexes(scales_hat_non_anchor)
                rv = decoder.decode_stream(index_non_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(z_hat.shape[0], -1, z_hat.shape[2]*2, z_hat.shape[3]*2)
                y_hat_slice_non_anchor = self.gaussian_conditional.dequantize(rv, means_hat_non_anchor)

                y_hat_slice = Multiplexer(y_hat_slice_non_anchor, y_hat_slice_anchor)
                y_hat_slices.append(y_hat_slice)

            else:
                support_slices = torch.cat([params] + [y_hat_slices[i] for i in range(slice_index)], dim=1)
                cc_params = self.cc_transforms[slice_index](support_slices)

                gaussian_params = self.entropy_parameters_4(
                    torch.cat((params, cc_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                
                index = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(z_hat.shape[0], -1, z_hat.shape[2]*4, z_hat.shape[3]*4)
                y_hat_slice = self.gaussian_conditional.dequantize(rv, means_hat)
                
                y_hat_slices.append(y_hat_slice)
        
        y_hat = torch.cat(y_hat_slices, dim=1)
        y_hat = y_hat * InterpolatedInverseGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat, _ = self.g_s(y_hat, quality, q_task)
        x_hat = x_hat.clamp_(0, 1)

        return {"x_hat": x_hat}
