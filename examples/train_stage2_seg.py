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
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from losses import RateDistortionLoss, compute_g_s_ratio_loss, compute_ratio
from ptflops import get_model_complexity_info

from compressai.zoo import image_models

sys.path.append(os.getcwd())
from seg_util import dataset, transform
from seg_util.util import intersectionAndUnionGPU


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
    base_dir = f'./checkpoints/train_stage2_seg/'
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


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad and (("g_s" in n and "g_s0" not in n) and ("score_predictor" in n or "fastmlp" in n))
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())

    # inter_params = parameters & aux_parameters
    # union_params = parameters | aux_parameters

    # assert len(inter_params) == 0
    # assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    logging.info(f'parameters in optimizer:')
    for n in sorted(parameters):
        logging.info(f'{n}')
    logging.info(f'parameters in aux_optimizer:')
    for n in sorted(aux_parameters):
        logging.info(f'{n}')
    return optimizer, aux_optimizer


def train_one_epoch(
    model,
    seg_model,
    seg_transforms,
    rd_criterion,
    train_dataloader,
    optimizer,
    aux_optimizer,
    epoch,
    clip_max_norm,
    args,
):
    model.train()
    device = next(model.parameters()).device
    lambda_list = [0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.0483, 0.0932, 0.18]
    beta = [0.0, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56]
    lambda_perc = 4.26 / 2.56

    for i, (samples, targets) in enumerate(train_dataloader):
        samples = samples.to(device)
        targets = targets.to(device)

        # aux_optimizer.zero_grad()

        q = random.randint(1, 8)
        q_task = random.randint(1, 8)
        out_net = model(samples, q, q_task, args.task_idx)

        lambda_rd = lambda_list[8 - q] * 100
        lambda_gan = beta[8 - 1]
        lambda_acc = 1.0
        g_s_ratio = 1 - (q_task - 1) / 7
        g_s_ratio_loss = compute_g_s_ratio_loss(out_net["decisions"], g_s_ratio=g_s_ratio)

        out_rd_criterion = rd_criterion(out_net, samples)

        out_net["x_hat"] = F.pad(out_net["x_hat"], (0, 1, 0, 1), mode="replicate")
        targets = F.pad(targets.float(), (0, 1, 0, 1), mode="replicate").long()
        if args.zoom_factor != 8:
            h = int((targets.size()[1] - 1) / 8 * args.zoom_factor + 1)
            w = int((targets.size()[2] - 1) / 8 * args.zoom_factor + 1)
            # 'nearest' mode doesn't support align_corners mode and 'bilinear' mode is fine for downsampling
            targets = F.interpolate(targets.unsqueeze(1).float(), size=(h, w), mode='bilinear', align_corners=True).squeeze(1).long()
        _, main_loss, aux_loss_ = seg_model(seg_transforms(out_net["x_hat"]), targets)
        main_loss, aux_loss_ = torch.mean(main_loss), torch.mean(aux_loss_)
        acc_loss = main_loss + args.aux_weight * aux_loss_

        optimizer.zero_grad()
        rd_loss = out_rd_criterion["bpp_loss"] * lambda_rd + out_rd_criterion["mse_loss"] / 100
        total_loss = rd_loss + lambda_perc * out_rd_criterion["lpips"] * lambda_gan + g_s_ratio_loss * 10.0 + acc_loss * lambda_acc
        total_loss.backward()
        aux_loss = model.aux_loss()
        # aux_loss.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

        optimizer.step()
        # aux_optimizer.step()

        if (i*len(samples) % 1000 == 0):
            logging.info(
                f'[{i*len(samples)}/{len(train_dataloader.dataset)}] (q={q})| '
                f'Loss: {total_loss.item():.4f} | '
                f'MSE loss: {out_rd_criterion["mse_loss"].item():.5f} | '
                f'LPIPS loss: {out_rd_criterion["lpips"].item():.5f} | '
                f'Bpp loss: {out_rd_criterion["bpp_loss"].item():.4f} | '
                f'Acc loss: {acc_loss.item():.4f} | '
                f'G Ratio loss ({g_s_ratio:.4f}): {g_s_ratio_loss.item():.4f} | '
                f"Aux loss: {aux_loss.item():.2f}"
            )


@torch.no_grad()
def test_epoch(epoch, test_dataloader, model, seg_model, seg_transforms, rd_criterion, acc_criterion, args):
    model.eval()
    device = next(model.parameters()).device
    lambda_list = [0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.0483, 0.0932, 0.18]
    beta = [0.0, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56]
    lambda_perc = 4.26 / 2.56

    total_loss = AverageMeter()
    total_bpp_loss = AverageMeter()
    total_mse_loss = AverageMeter()
    total_lpips_loss = AverageMeter()
    total_acc_loss = AverageMeter()
    total_aux_loss = AverageMeter()
    total_mIoU = AverageMeter()
    total_mAcc = AverageMeter()
    total_allAcc = AverageMeter()
    lambda_gan = beta[8 - 1]
    lambda_acc = 1.0

    for q in range(1, 9):
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()
        lpips_loss = AverageMeter()
        acc_loss = AverageMeter()
        aux_loss = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        ratio = AverageMeter()
        lambda_rd = lambda_list[8 - q] * 100

        with torch.no_grad():
            for batch in test_dataloader:
                images = batch[0]
                target = batch[-1]
                images = images.to(device)
                target = target.to(device)

                out_net = model(images, q, q_task=8, task_idx=args.task_idx)
                out_rd_criterion = rd_criterion(out_net, images)

                out_net["x_hat"] = F.pad(out_net["x_hat"], (0, 1, 0, 1), mode="replicate")
                target = F.pad(target.float(), (0, 1, 0, 1), mode="replicate").long()
                output, _, _ = seg_model(seg_transforms(out_net["x_hat"]), target)
                if args.zoom_factor != 8:
                    output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
                acc = acc_criterion(output, target)
                acc = torch.mean(acc)

                output = output.max(1)[1]
                intersection, union, target = intersectionAndUnionGPU(output, target, args.nb_classes, args.ignore_label)
                intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

                rd_loss = out_rd_criterion["bpp_loss"] * lambda_rd + out_rd_criterion["mse_loss"] / 100
                egd_loss = rd_loss + lambda_perc * out_rd_criterion["lpips"] * lambda_gan + acc * lambda_acc

                loss.update(egd_loss, images.size(0))
                bpp_loss.update(out_rd_criterion["bpp_loss"], images.size(0))
                mse_loss.update(out_rd_criterion["mse_loss"], images.size(0))
                lpips_loss.update(out_rd_criterion["lpips"], images.size(0))
                acc_loss.update(acc, images.size(0))
                aux_loss.update(model.aux_loss(), images.size(0))
                ratio.update(compute_ratio(out_net["decisions"]), images.size(0))

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        logging.info(
            f"Test epoch {epoch}: Average losses (q={q}): "
            f"Loss: {loss.avg:.4f} | "
            f"MSE loss: {mse_loss.avg:.5f} | "
            f"LPIPS loss: {lpips_loss.avg:.5f} | "
            f"Bpp loss: {bpp_loss.avg:.4f} | "
            f"Acc loss: {acc_loss.avg:.4f} | "
            f"mIoU: {mIoU:.4f} | "
            f"mAcc: {mAcc:.4f} | "
            f"allAcc: {allAcc:.4f} | "
            f"Ratio: {ratio.avg:.4f} | "
            f"Aux loss: {aux_loss.avg:.2f}"
        )

        total_loss.update(loss.avg)
        total_mse_loss.update(mse_loss.avg)
        total_lpips_loss.update(lpips_loss.avg)
        total_bpp_loss.update(bpp_loss.avg)
        total_acc_loss.update(acc_loss.avg)
        total_aux_loss.update(aux_loss.avg)
        total_mIoU.update(mIoU)
        total_mAcc.update(mAcc)
        total_allAcc.update(allAcc)

    logging.info(
        f"Test epoch {epoch}: Average losses (total): "
        f"Loss: {total_loss.avg:.4f} | "
        f"MSE loss: {total_mse_loss.avg:.5f} | "
        f"LPIPS loss: {total_lpips_loss.avg:.5f} | "
        f"Bpp loss: {total_bpp_loss.avg:.4f} | "
        f"Acc loss: {total_acc_loss.avg:.4f} | "
        f"mIoU: {total_mIoU.avg:.4f} | "
        f"mAcc: {total_mAcc.avg:.4f} | "
        f"allAcc: {total_allAcc.avg:.4f} | "
        f"Aux loss: {total_aux_loss.avg:.2f}\n"
    )

    return total_loss.avg, total_mIoU.avg


def save_checkpoint(state, is_best, base_dir, iter=None, filename="checkpoint.pth.tar"):
    torch.save(state, base_dir+filename) if iter is None else torch.save(state, base_dir+f"iter{iter}_"+filename)
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
        default=2,
        type=int,
        help="Task index (0: MSE, 1: Cls, 2: Seg)",
    )

    # Segmentation parameters
    parser.add_argument(
        "-a",
        "--arch",
        default="psp",
        choices=["psp", "psa"],
        help="Segmentation network architecture (default: %(default)s)",
    )
    parser.add_argument('-d', '--data_path', default='dataset/ade20k', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=150, type=int,
                        help='number of the classification types')
    parser.add_argument('--layers', default=50, type=int,
                        help='number of layers')
    parser.add_argument('--scale_min', default=0.5, type=float,
                        help='minimum random scale')
    parser.add_argument('--scale_max', default=2.0, type=float,
                        help='maximum random scale')
    parser.add_argument('--rotate_min', default=-10, type=int,
                        help='minimum random rotate')
    parser.add_argument('--rotate_max', default=10, type=int,
                        help='maximum random rotate')
    parser.add_argument('--zoom_factor', default=8, type=int,
                        help='zoom factor for final prediction during training, be in [1, 2, 4, 8]')
    parser.add_argument('--ignore_label', default=255, type=int,
                        help='ignore_label')
    parser.add_argument('--aux_weight', default=0.4, type=float,
                        help='aux_weight')
    parser.add_argument('--psa_type', default=2, type=int,
                        help='0-collect, 1-distribute, 2-bi-direction')
    parser.add_argument('--compact', default=0, type=int,
                        help='0-no, 1-yes')
    parser.add_argument('--shrink_factor', default=2, type=int,
                        help='shrink factor when get attention mask')
    parser.add_argument('--mask_h', type=int, help='specify mask h or not')
    parser.add_argument('--mask_w', type=int, help='specify mask w or not')
    parser.add_argument('--normalization_factor', default=1.0, type=float,
                        help='normalization factor for aggregation')
    parser.add_argument('--psa_softmax', default=1, type=int,
                        help='softmax on mask or not: 0-no, 1-yes')
    parser.add_argument('--model_path', default='checkpoints/pspnet/pspnet_train_epoch_100.pth', type=str,
                        help='evaluation model path')
    parser.add_argument('--colors_path', default='seg_data/ade20k/ade20k_colors.txt', type=str,
                        help='path of dataset colors')
    parser.add_argument('--names_path', default='seg_data/ade20k/ade20k_names.txt', type=str,
                        help='path of dataset category names')

    parser.add_argument(
        "-e",
        "--epochs",
        default=200,
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
        "-lrd",
        "--learning_rate_d",
        default=1e-4,
        type=float,
        help="Learning rate for discriminator (default: %(default)s)",
    )
    parser.add_argument(
        "--milestones",
        nargs="+",
        type=int,
        default=(150,),
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
    args = parser.parse_args(argv)
    return args


def check(args):
    assert args.nb_classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    if args.arch == 'psp':
        # assert (args.patch_size - 1) % 8 == 0 and (args.patch_size - 1) % 8 == 0
        assert (args.patch_size + 1 - 1) % 8 == 0 and (args.patch_size + 1 - 1) % 8 == 0
    elif args.arch == 'psa':
        if args.compact:
            args.mask_h = (args.patch_size - 1) // (8 * args.shrink_factor) + 1
            args.mask_w = (args.patch_size - 1) // (8 * args.shrink_factor) + 1
        else:
            assert (args.mask_h is None and args.mask_w is None) or (
                        args.mask_h is not None and args.mask_w is not None)
            if args.mask_h is None and args.mask_w is None:
                args.mask_h = 2 * ((args.patch_size - 1) // (8 * args.shrink_factor) + 1) - 1
                args.mask_w = 2 * ((args.patch_size - 1) // (8 * args.shrink_factor) + 1) - 1
            else:
                assert (args.mask_h % 2 == 1) and (args.mask_h >= 3) and (
                        args.mask_h <= 2 * ((args.patch_size - 1) // (8 * args.shrink_factor) + 1) - 1)
                assert (args.mask_w % 2 == 1) and (args.mask_w >= 3) and (
                        args.mask_w <= 2 * ((args.patch_size - 1) // (8 * args.shrink_factor) + 1) - 1)
    else:
        raise Exception('architecture not supported yet'.format(args.arch))


def main(argv):
    args = parse_args(argv)
    check(args)
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

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean_scaled = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std_scaled = [item * value_scale for item in std]

    train_transform = transform.Compose([
        transform.RandScale([args.scale_min, args.scale_max]),
        # transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean_scaled, ignore_label=args.ignore_label),
        # transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.patch_size, args.patch_size], crop_type='rand', padding=mean_scaled, ignore_label=args.ignore_label),
        transform.ToTensor(),
        # transform.Normalize(mean=mean, std=std),
    ])
    train_dataset = dataset.SemData(split='train', data_root=args.data_path, data_list=f"{args.data_path}/list/training.txt", transform=train_transform)

    test_transform = transform.Compose([
        transform.Resize(args.patch_size, max_size=2048),
        transform.Crop([args.patch_size, args.patch_size], crop_type='center', padding=mean_scaled, ignore_label=args.ignore_label),
        transform.ToTensor(),
        # transform.Normalize(mean=mean, std=std),
    ])
    test_dataset = dataset.SemData(split='val', data_root=args.data_path, data_list=f"{args.data_path}/list/validation.txt", transform=test_transform)

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

    seg_transforms = [
        # transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    seg_transforms = transforms.Compose(seg_transforms)
    acc_criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    if args.arch == 'psp':
        from seg_model.pspnet import PSPNet
        seg_model = PSPNet(layers=args.layers, classes=args.nb_classes, zoom_factor=args.zoom_factor, criterion=acc_criterion, pretrained=False)
    elif args.arch == 'psa':
        from seg_model.psanet import PSANet
        seg_model = PSANet(layers=args.layers, classes=args.nb_classes, zoom_factor=args.zoom_factor, psa_type=args.psa_type,
                       compact=args.compact, shrink_factor=args.shrink_factor, mask_h=args.mask_h, mask_w=args.mask_w,
                       normalization_factor=args.normalization_factor, psa_softmax=args.psa_softmax, criterion=acc_criterion, pretrained=False)
    seg_model.to(device)
    seg_checkpoint = torch.load(args.model_path)['state_dict']
    seg_checkpoint = {k.replace("module.", ""): v for k, v in seg_checkpoint.items()}
    seg_model.load_state_dict(seg_checkpoint)
    seg_model.eval()

    rd_criterion = RateDistortionLoss().to(device).eval()
    acc_criterion.to(device).eval()

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
        seg_model = CustomDataParallel(seg_model)

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

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=args.milestones,
                                                  gamma=args.gamma)

    last_epoch = 0
    best_loss = float("inf")
    best_miou = float("-inf")
    if args.checkpoint:  # load from previous checkpoint
        logging.info("Loading "+str(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["loss"]
        best_miou = checkpoint["miou"]
        net.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    for epoch in range(last_epoch, args.epochs):
        logging.info('======Current epoch %s ======'%epoch)
        logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            seg_model,
            seg_transforms,
            rd_criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            args,
        )
        loss, miou = test_epoch(epoch, test_dataloader, net, seg_model, seg_transforms, rd_criterion, acc_criterion, args)
        lr_scheduler.step()

        best_loss = min(loss, best_loss)
        is_best = (miou >= best_miou)
        best_miou = max(miou, best_miou)

        if is_best:
            logging.info(f'Save epoch {epoch} as the best')

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": best_loss,
                    "miou": best_miou,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                base_dir,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
