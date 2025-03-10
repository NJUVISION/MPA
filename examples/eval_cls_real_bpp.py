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
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data.mixup import Mixup
from timm import create_model
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from losses import compute_ratio_loss, compute_ratio
from ptflops import get_model_complexity_info

from compressai.datasets.classification import build_dataset
from compressai.zoo import image_models


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


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


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_list = [0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.0483, 0.0932, 0.18]

    def forward(self, output, target, quality):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lambda_list[quality-1] * 255**2 * out["mse_loss"] + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def init(args):
    base_dir = f'./checkpoints/{args.model}_eval/cls/{args.cls_model}/task_idx_{args.task_idx}/q_dec_{args.q_task}/'
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


def cal_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    psnr = []
    for i in range(len(a)):
        mse = F.mse_loss(a[i], b[i]).item()
        psnr.append(-10 * math.log10(mse))
    return sum(psnr) / len(psnr)


@torch.no_grad()
def test_epoch(epoch, test_dataloader, model, cls_model, cls_transforms, rd_criterion, q_task, args):
    acc_criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    device = next(model.parameters()).device
    lambda_list = [0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.0483, 0.0932, 0.18]

    total_bpp = AverageMeter()
    total_psnr = AverageMeter()
    total_acc_loss = AverageMeter()
    total_acc1 = AverageMeter()
    total_acc5 = AverageMeter()
    total_aux_loss = AverageMeter()

    for q in range(1, 9):
        bpp = AverageMeter()
        psnr = AverageMeter()
        acc_loss = AverageMeter()
        acc1 = AverageMeter()
        acc5 = AverageMeter()
        aux_loss = AverageMeter()

        with torch.no_grad():
            for batch in test_dataloader:
                images = batch[0]
                target = batch[-1]
                images = images.to(device)
                target = target.to(device)

                if args.real_bpp:
                    out_enc = model.compress(images, q)
                    out_dec = model.decompress(out_enc["strings"], out_enc["shape"], q, q_task=q_task, task_idx=args.task_idx)

                    num_pixels = images.size(0) * images.size(2) * images.size(3)
                    real_bpp = sum(sum(len(ss) for ss in s) for s in out_enc["strings"]) * 8.0 / num_pixels
                else:
                    out_dec = model(images, quality=q, q_task=q_task, task_idx=args.task_idx)
                    num_pixels = images.size(0) * images.size(2) * images.size(3)
                    real_bpp = sum(
                        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                        for likelihoods in out_dec["likelihoods"].values()
                    )
                out_dec['x_hat'] = out_dec['x_hat'].clamp_(0, 1)

                cls = cls_model(cls_transforms(out_dec["x_hat"]))
                acc = acc_criterion(cls, target)
                top1, top5 = accuracy(cls, target, topk=(1, 5))
                bpp.update(real_bpp, images.size(0))
                psnr.update(cal_psnr(images, out_dec["x_hat"]), images.size(0))
                acc_loss.update(acc, images.size(0))
                acc1.update(top1, images.size(0))
                acc5.update(top5, images.size(0))
                aux_loss.update(model.aux_loss(), images.size(0))

        logging.info(
            f"Test epoch {epoch}: Average losses (q={q}): "
            f"PSNR: {psnr.avg:.5f} | "
            f"Bpp: {bpp.avg:.4f} | "
            f"Acc loss: {acc_loss.avg:.4f} | "
            f"Acc: Top1({acc1.avg:.4f}), Top5({acc5.avg:.4f}) | "
            f"Aux loss: {aux_loss.avg:.2f}"
        )

        total_psnr.update(psnr.avg)
        total_bpp.update(bpp.avg)
        total_acc_loss.update(acc_loss.avg)
        total_acc1.update(acc1.avg)
        total_acc5.update(acc5.avg)
        total_aux_loss.update(aux_loss.avg)

    logging.info(
        f"Test epoch {epoch}: Average losses (total): "
        f"PSNR: {total_psnr.avg:.5f} | "
        f"Bpp: {total_bpp.avg:.4f} | "
        f"Acc loss: {total_acc_loss.avg:.4f} | "
        f"Acc: Top1({total_acc1.avg:.4f}), Top5({total_acc5.avg:.4f}) | "
        f"Aux loss: {total_aux_loss.avg:.2f}\n"
    )

    return total_acc_loss.avg


def save_checkpoint(state, is_best, base_dir, is_iter=False, filename="checkpoint.pth.tar"):
    torch.save(state, base_dir+filename) if not is_iter else torch.save(state, base_dir+"iter_"+filename)
    if is_best:
        shutil.copyfile(base_dir+filename, base_dir+"checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="mpa",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-i",
        "--task_idx",
        default=1,
        type=int,
        help="Task index (0: MSE, 1: Cls, 2: Seg)",
    )
    parser.add_argument(
        "--cls_model",
        default="convnext_tiny",
        choices=["convnext_tiny", "swinv2_tiny_window8_256", "tv_resnet50"],
        help="CLS Model architecture (default: %(default)s)",
    )

    # Dataset parameters
    parser.add_argument('-d', '--data_path', default='/workspace/shared/ImageNet-1K', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'IMNET100', 'image_folder'],
                        type=str, help='ImageNet dataset path')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    parser.add_argument(
        "-e",
        "--epochs",
        default=4,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--milestones",
        nargs="+",
        type=int,
        default=(3,),
        help="Milestones (default: %(default)s)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Multiplicative factor of learning rate decay (default: %(default)s)",
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
        default=3,
        help="Quality level (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=16,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
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
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=int, default=10086, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        '--name', 
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), 
        type=str,
        help='Result dir name', 
    )
    parser.add_argument("--pretrained", type=str, help="Path to a pretrained model")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--q_task", default=8.0, type=float, help="q_task value")
    parser.add_argument("--real_bpp", action="store_true", help="real bpp")
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
        np.random.seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed_all(args.seed)
    
    setup_logger(base_dir + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    msg = f'======================= {args.name} ======================='
    logging.info(msg)
    for k in args.__dict__:
        logging.info(k + ':' + str(args.__dict__[k]))
    logging.info('=' * len(msg))

    scale = (0.08, 1.0)  # default imagenet scale range
    ratio = (3./4., 4./3.)  # default imagenet ratio range
    train_transforms = transforms.Compose([
        RandomResizedCropAndInterpolation(args.patch_size, scale=scale, ratio=ratio, interpolation='random'),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor()
        ]
    )

    t = []
    # warping (no cropping) when evaluated at 384 or larger
    crop_pct = 224 / 256
    size = int(args.patch_size / crop_pct)
    t.append(
        # to maintain same ratio w.r.t. 224 images
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
    )
    t.append(transforms.CenterCrop(args.patch_size))
    t.append(transforms.ToTensor())
    test_transforms = transforms.Compose(t)

    train_dataset, args.nb_classes = build_dataset(is_train=True, transform=train_transforms, args=args)
    test_dataset, _ = build_dataset(is_train=False, transform=test_transforms, args=args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = image_models[args.model](quality=int(args.quality_level))
    net = net.to(device)

    cls_transforms = [
        # transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.CenterCrop(args.patch_size),
        # transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ]
    cls_transforms = transforms.Compose(cls_transforms)
    cls_model = create_model(args.cls_model, pretrained=True).to(device).eval()

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
        cls_model = CustomDataParallel(cls_model)

    if args.pretrained:  # load pretrained model
        logging.info("Loading "+str(args.pretrained))
        try:
            pretrained = torch.load(args.pretrained, map_location=device)["state_dict"]
            logging.info(net.load_state_dict(pretrained, strict=False))
            net.update(force=True)
        except KeyError:
            pretrained = torch.load(args.pretrained, map_location=device)
            logging.info(net.load_state_dict(pretrained, strict=False))

    macs, params = get_model_complexity_info(net.eval(), (3, 256, 256), as_strings=False, print_per_layer_stat=False)
    logging.info("MACs/pixel:"+str(macs/(256**2)))
    logging.info("params:"+str(params))

    rd_criterion = RateDistortionLoss()
    epoch = 0
    test_epoch(epoch, test_dataloader, net, cls_model, cls_transforms, rd_criterion, args.q_task, args)


if __name__ == "__main__":
    main(sys.argv[1:])
