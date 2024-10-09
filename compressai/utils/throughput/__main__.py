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

import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd

from torch.utils.data import DataLoader
from torchvision import transforms

from ptflops import get_model_complexity_info

from compressai.datasets import ImageFolder
from compressai.zoo import image_models


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def init(args):
    base_dir = f'./latency/{args.architecture}/{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)

    return base_dir


def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)


@torch.no_grad()
def test_throughput(d, model, args):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for _ in range(args.warm_up):
            _ = model(d, args.quality_level, args.q_task)    # warm up

        # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, with_modules=True, profile_memory=True) as prof:
        logging.info("Epoch start!")
        start = time.time()
        for _ in range(args.avg_times):
            _ = model(d, args.quality_level, args.q_task)
        forward_time = (time.time() - start) / (args.avg_times * len(d))
        # logging.info("\n"+prof.key_averages().table())
        throughput = 1 / forward_time
        logging.info("Epoch end!")

    logging.info(
        f"forward_time: {forward_time} s / image | "
        f"throughput: {throughput:.3f} images / s"
    )

    return forward_time, throughput


@torch.no_grad()
def test_throughput_ec(d, model, args):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for _ in range(args.warm_up):
            out_enc = model.compress(d, args.quality_level) # warm up
            _ = model.decompress(out_enc["strings"], out_enc["shape"], args.quality_level, args.q_task)

        logging.info("Epoch start!")
        start = time.time()
        for _ in range(args.avg_times):
            out_enc = model.compress(d, args.quality_level)
        enc_time = (time.time() - start) / (args.avg_times * len(d))
        enc_throughput = 1 / enc_time

        start = time.time()
        for _ in range(args.avg_times):
            _ = model.decompress(out_enc["strings"], out_enc["shape"], args.quality_level, args.q_task)
        dec_time = (time.time() - start) / (args.avg_times * len(d))
        dec_throughput = 1 / dec_time
        logging.info("Epoch end!")

    logging.info(
        f"enc_time: {enc_time} s / image | "
        f"dec_time: {dec_time} s / image | "
        f"enc_throughput: {enc_throughput:.3f} images / s | "
        f"dec_throughput: {dec_throughput:.3f} images / s"
    )

    return enc_time, dec_time, enc_throughput, dec_throughput


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-a",
        "--architecture",
        default="tinylic",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=1,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num_workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--quality_level",
        type=int,
        default=8,
        help="Quality level (default: %(default)s)",
    )
    parser.add_argument(
        "--q_task",
        type=int,
        default=1,
        help="Quality level (default: %(default)s)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=16,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--warm_up",
        type=int,
        default=10,
        help="Warm up times (default: %(default)s)",
    )
    parser.add_argument(
        "--avg_times",
        type=int,
        default=1000,
        help="Average times (default: %(default)s)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--gpu_id",
        type=str,
        default=0,
        help="GPU ids (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        '--name', 
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), 
        type=str,
        help='Result dir name', 
    )
    parser.add_argument("--pretrained", type=str, help="Path to a pretrained model")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("-ec", "--entropy_coding", action="store_true", help="Enable entropy coding")
    args = parser.parse_args(argv)
    return args


def rename_key(key: str) -> str:
    """Rename state_dict key."""
    if "score_predictor" in key:
        return f"skip"
    return key


def modify_value(key, value):
    if "gamma" in key:
        return torch.abs(value * 10.0)
    return value


def main(argv):
    args = parse_args(argv)
    base_dir = init(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    setup_logger(base_dir + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    msg = f'======================= {args.name} ======================='
    logging.info(msg)
    for k in args.__dict__:
        logging.info(k + ':' + str(args.__dict__[k]))
    logging.info('=' * len(msg))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    net = image_models[args.architecture](quality=int(args.quality_level))
    net = net.to(device)
    
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    if args.pretrained:  # load pretrained model
        logging.info("Loading "+str(args.pretrained))
        try:
            pretrained = torch.load(args.pretrained, map_location=device)["state_dict"]
            net.load_state_dict(pretrained, strict=False)
            net.update(force=True)
        except KeyError:
            pretrained = torch.load(args.pretrained, map_location=device)
            net.load_state_dict(pretrained, strict=False)

    macs, params = get_model_complexity_info(net.eval(), (3, 256, 256), as_strings=False, print_per_layer_stat=False)
    logging.info("MACs/pixel:"+str(macs/(256**2)))
    logging.info("params:"+str(params))

    if args.checkpoint:  # load from previous checkpoint
        logging.info("Loading "+str(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"], strict=True)

    d = torch.zeros(args.test_batch_size, 3, args.patch_size, args.patch_size, device=device)
    if args.entropy_coding:
        for _ in range(args.epochs):
            _, _, _, _ = test_throughput_ec(d, net, args)
    else:
        for _ in range(args.epochs):
            _, _ = test_throughput(d, net, args)

if __name__ == "__main__":
    main(sys.argv[1:])
