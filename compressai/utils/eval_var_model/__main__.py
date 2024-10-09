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
"""
Evaluate an end-to-end compression model on an image dataset.
"""
import argparse
import json
import math
import os
import sys
import time
import numpy as np

from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pytorch_msssim import ms_ssim
import lpips
from torchvision import transforms

import compressai

from compressai.zoo import image_models as pretrained_models
from compressai.zoo import load_state_dict
from compressai.zoo.image import model_architectures as architectures

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    # t = []
    # # warping (no cropping) when evaluated at 384 or larger
    # crop_pct = 224 / 256
    # size = int(256 / crop_pct)
    # t.append(
    #     # to maintain same ratio w.r.t. 224 images
    #     transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
    # )
    # t.append(transforms.CenterCrop(256))
    # t.append(transforms.ToTensor())
    # test_transforms = transforms.Compose(t)
    filename = os.path.split(filepath)[-1]
    return transforms.ToTensor()(img), filename
    # return test_transforms(img) # transforms.ToTensor()(img)


@torch.no_grad()
def inference(model, x, lpips_metric, q=1, q_task=1, filename=None, args=None):
    x = x.unsqueeze(0)

    h, w = x.size(2), x.size(3)
    p = 256  # maximum 6 strides of 2, and window size 4 for the smallest latent fmap: 4*2^6=256
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    start = time.time()
    out_enc = model.compress(x_padded, q)
    enc_time = time.time() - start
    enc_mem = torch.cuda.max_memory_allocated()

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"], q, q_task, args.task_idx)
    dec_time = time.time() - start
    dec_mem = torch.cuda.max_memory_allocated()

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    if args.save:
        filename = os.path.splitext(filename)[0]
        rec_img = transforms.ToPILImage()(out_dec['x_hat'].clamp_(0, 1).squeeze().cpu())
        rec_img.save(os.path.join(args.save, f'q={q}', f'q_task={q_task}', f'{filename}.png'))

    return {
        # "quality": q,
        "psnr": psnr(x, out_dec["x_hat"].clamp_(0, 1)),
        "ms-ssim": ms_ssim(x, out_dec["x_hat"].clamp_(0, 1), data_range=1.0).item(),
        "lpips": lpips_metric(x, out_dec["x_hat"].clamp_(0, 1), normalize=True).item(),
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
        "enc_mem": enc_mem,
        "dec_mem": dec_mem,
    }


@torch.no_grad()
def inference_entropy_estimation(model, x, lpips_metric, q=1, q_task=1, filename=None, args=None):
    x = x.unsqueeze(0)

    h, w = x.size(2), x.size(3)
    p = 256  # maximum 6 strides of 2, and window size 4 for the smallest latent fmap: 4*2^6=256
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    start = time.time()
    out_net = model.forward(x_padded, q, q_task, args.task_idx)
    elapsed_time = time.time() - start

    out_net["x_hat"] = F.pad(
        out_net["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )

    if args.save:
        filename = os.path.splitext(filename)[0]
        rec_img = transforms.ToPILImage()(out_net['x_hat'].clamp_(0, 1).squeeze().cpu())
        rec_img.save(os.path.join(args.save, f'q={q}', f'q_task={q_task}', f'{filename}.png'))

    return {
        # "quality": q,
        "psnr": psnr(x, out_net["x_hat"].clamp_(0, 1)),
        "ms-ssim": ms_ssim(x, out_net["x_hat"].clamp_(0, 1), data_range=1.0).item(),
        "lpips": lpips_metric(x, out_net["x_hat"].clamp_(0, 1), normalize=True).item(),
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def load_pretrained(model: str, metric: str, quality: int) -> nn.Module:
    return pretrained_models[model](
        quality=quality, metric=metric, pretrained=True
    ).eval()


def load_checkpoint(arch: str, checkpoint_path: str) -> nn.Module:
    try:
        state_dict = load_state_dict(torch.load(checkpoint_path)['state_dict'])
    except KeyError:
        state_dict = load_state_dict(torch.load(checkpoint_path))
    return architectures[arch].from_state_dict(state_dict).eval()


def eval_model(model, filepaths, lpips_metric, entropy_estimation=False, half=False, q=1, q_task=1, args=None):
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    for f in filepaths:
        x, filename = read_image(f)
        x = x.to(device)
        if not entropy_estimation:
            if half:
                model = model.half()
                x = x.half()
            rv = inference(model, x, lpips_metric, q, q_task, filename, args)
        else:
            rv = inference_entropy_estimation(model, x, lpips_metric, q, q_task, filename, args)
        print(json.dumps(rv, indent=2))
        if args.save:
            filename = f'{os.path.splitext(filename)[0]}.json'
            with open(os.path.join(args.save, f'q={q}', f'q_task={q_task}', filename), 'w') as file:
                json.dump(rv, file, indent=2)
        for k, v in rv.items():
            metrics[k] += v
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    if args.save:
        with open(os.path.join(args.save, f'q={q}', f'results_q{q}_q_task{q_task}.json'), 'w') as file:
            json.dump(metrics, file, indent=2)
    return metrics


def setup_args():
    parent_parser = argparse.ArgumentParser(
        add_help=False,
    )

    # Common options.
    parent_parser.add_argument("dataset", type=str, help="dataset path")
    parent_parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        choices=pretrained_models.keys(),
        help="model architecture",
        required=True,
    )
    parent_parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parent_parser.add_argument(
        "--cuda",
        action="store_true",
        help="enable CUDA",
    )
    parent_parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parent_parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    parent_parser.add_argument(
        "--q_task",
        type=float,
        default=1.0,
        help="q_task",
    )
    parent_parser.add_argument(
        "--task_idx",
        default=0,
        type=int,
        help="Task index (0: MSE, 1: Cls, 2: Seg)",
    )
    parent_parser.add_argument(
        "--save",
        type=str,
        help="Path to save dir"
    )

    parser = argparse.ArgumentParser(
        description="Evaluate a model on an image dataset.", add_help=True
    )
    subparsers = parser.add_subparsers(help="model source", dest="source")

    # Options for pretrained models
    pretrained_parser = subparsers.add_parser("pretrained", parents=[parent_parser])
    pretrained_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["mse", "ms-ssim"],
        default="mse",
        help="metric trained against (default: %(default)s)",
    )
    pretrained_parser.add_argument(
        "-q",
        "--quality",
        dest="qualities",
        nargs="+",
        type=int,
        default=(1,),
    )

    checkpoint_parser = subparsers.add_parser("checkpoint", parents=[parent_parser])
    checkpoint_parser.add_argument(
        "-p",
        "--path",
        dest="paths",
        type=str,
        nargs="*",
        required=True,
        help="checkpoint path",
    )

    return parser


def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)

    if args.save:
        for q in range(1, 9):
            os.makedirs(os.path.join(args.save, f'q={q}', f'q_task={args.q_task}'), exist_ok=True)

    if not args.source:
        print("Error: missing 'checkpoint' or 'pretrained' source.", file=sys.stderr)
        parser.print_help()
        raise SystemExit(1)

    filepaths = collect_images(args.dataset)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        raise SystemExit(1)

    compressai.set_entropy_coder(args.entropy_coder)

    if args.source == "pretrained":
        runs = sorted(args.qualities)
        opts = (args.architecture, args.metric)
        load_func = load_pretrained
        log_fmt = "\rEvaluating {0} | {run:d}"
    elif args.source == "checkpoint":
        runs = args.paths
        opts = (args.architecture,)
        load_func = load_checkpoint
        log_fmt = "\rEvaluating {run:s}"

    lpips_metric = lpips.LPIPS(net='alex')
    if args.cuda and torch.cuda.is_available():
        lpips_metric = lpips_metric.to("cuda")

    results = defaultdict(list)
    for run in runs:
        model = load_func(*opts, run)
        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")

        if args.source == "checkpoint":
            model.update(force=True)

        for q in range(1, 9):
            if args.verbose:
                sys.stderr.write(log_fmt.format(*opts, run=run))
                sys.stderr.flush()

            metrics = eval_model(model, filepaths, lpips_metric, args.entropy_estimation, args.half, q, args.q_task, args)
            for k, v in metrics.items():
                results[k].append(v)

    if args.verbose:
        sys.stderr.write("\n")
        sys.stderr.flush()

    description = (
        "entropy estimation" if args.entropy_estimation else args.entropy_coder
    )
    output = {
        "name": f"{args.architecture} (q_task = {args.q_task})",
        "description": f"Inference ({description})",
        "results": results,
    }

    print(json.dumps(output, indent=2))
    if args.save:
        filename = f'results_q_task{args.q_task}.json'
        with open(os.path.join(args.save, filename), 'w') as file:
            json.dump(output, file, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])
