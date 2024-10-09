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

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import random
import lpips


def compute_ratio_loss(decisions, ratio, g_s_ratio=None):
    pred_loss = 0.0
    dec_pred_loss = 0.0
    enc_pred_score = decisions['enc']
    dec_pred_score = decisions['dec']
    henc_pred_score = decisions['henc']
    hdec_pred_score = decisions['hdec']
    if g_s_ratio is None:
        pred_score = [*enc_pred_score, *dec_pred_score, *henc_pred_score, *hdec_pred_score]
    else:
        pred_score = [*enc_pred_score, *henc_pred_score, *hdec_pred_score]

    for i, score in enumerate(pred_score):
        pos_ratio = score.mean(dim=(1, 2))
        pred_loss = pred_loss + ((pos_ratio - ratio) ** 2).mean()

    ratio_loss = pred_loss / len(pred_score)

    if g_s_ratio is not None:
        for i, score in enumerate(dec_pred_score):
            pos_ratio = score.mean(dim=(1, 2))
            dec_pred_loss = dec_pred_loss + ((pos_ratio - g_s_ratio) ** 2).mean()
        
        dec_ratio_loss = dec_pred_loss / len(dec_pred_score)

    return ratio_loss if g_s_ratio is None else (ratio_loss, dec_ratio_loss)


def compute_g_a_ratio_loss(decisions, g_a_ratio=None):
    enc_pred_loss = 0.0
    enc_pred_score = decisions['enc']

    if g_a_ratio is not None:
        for i, score in enumerate(enc_pred_score):
            pos_ratio = score.mean(dim=(1,2))
            enc_pred_loss = enc_pred_loss + ((pos_ratio - g_a_ratio) ** 2).mean()
        
        enc_ratio_loss = enc_pred_loss / len(enc_pred_score)

    return enc_ratio_loss


def compute_g_s_ratio_loss(decisions, g_s_ratio=None):
    dec_pred_loss = 0.0
    dec_pred_score = decisions['dec']

    if g_s_ratio is not None:
        for i, score in enumerate(dec_pred_score):
            pos_ratio = score.mean(dim=(1,2))
            dec_pred_loss = dec_pred_loss + ((pos_ratio - g_s_ratio) ** 2).mean()
        
        dec_ratio_loss = dec_pred_loss / len(dec_pred_score)

    return dec_ratio_loss


def compute_ratio(decisions):
    num_elements = 0
    num_nonzeros = 0
    enc_pred_score = decisions['enc']
    dec_pred_score = decisions['dec']
    henc_pred_score = decisions['henc']
    hdec_pred_score = decisions['hdec']
    pred_score = [*enc_pred_score, *dec_pred_score, *henc_pred_score, *hdec_pred_score]

    for i, score in enumerate(pred_score):
        if len(score) == 2:
            num_elements += (score[0].flatten(0).shape[0] + score[1].flatten(0).shape[0])
            num_nonzeros += score[0].flatten(0).shape[0]
        else:
            score = score.flatten(0)
            num_elements += score.shape[0]
            num_nonzeros += torch.count_nonzero(score)

    ratio = float(num_nonzeros / num_elements)

    return ratio


def padding_and_trimming(
    x_rec, # decoder output
    x # reference image
):
    _, _, H, W = x.size()
    x_rec = F.pad(x_rec, (15, 15, 15, 15), mode='replicate')
    x = F.pad(x, (15, 15, 15, 15), mode='replicate')
    _, _, H_pad, W_pad = x.size()
    top = random.randrange(0, 16)
    bottom = H_pad - random.randrange(0, 16)
    left = random.randrange(0, 16)
    right = W_pad - random.randrange(0, 16)
    x_rec = F.interpolate(x_rec[:, :, top:bottom, left:right],
    size=(H, W), mode='bicubic', align_corners=False)
    x = F.interpolate(x[:, :, top:bottom, left:right],
    size=(H, W), mode='bicubic', align_corners=False)
    return x_rec, x


class RateDistortionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips = lpips.LPIPS(net='vgg')
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        if self.training:
            out["lpips"] = self.lpips(*padding_and_trimming(output["x_hat"], target), normalize=True)
        else:
            out["lpips"] = self.lpips(output["x_hat"], target, normalize=True)

        out["lpips"] = torch.mean(out["lpips"])

        out["mse_loss"] = torch.mean(self.mse(255 * output['x_hat'], 255 * target))

        return out


class GANLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self,D_real_logits, D_gen_logits,mode='generator_loss'):
        D_loss_real = F.binary_cross_entropy_with_logits(input=D_real_logits,target=torch.ones_like(D_real_logits))
        D_loss_gen = F.binary_cross_entropy_with_logits(input=D_gen_logits,target=torch.zeros_like(D_gen_logits))
        D_loss = D_loss_real + D_loss_gen
        G_loss = F.binary_cross_entropy_with_logits(input=D_gen_logits,target=torch.ones_like(D_gen_logits))
        loss = G_loss if mode == 'generator_loss' else D_loss
        return loss
