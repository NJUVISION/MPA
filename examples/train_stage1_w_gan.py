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

from torch.utils.data import DataLoader
from torchvision import transforms

from losses import RateDistortionLoss, GANLoss, compute_ratio, compute_g_a_ratio_loss
from ptflops import get_model_complexity_info

from compressai.datasets import ImageFolder
from compressai.zoo import image_models
from compressai.models.discriminator import init_weights, DiscriminatorHiFiC_Independent


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
    base_dir = f'./checkpoints/train_stage1_w_gan/'
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
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())

    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

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
    model, model_disc, criterion, train_dataloader, optimizer, optimizer_D, aux_optimizer, epoch, clip_max_norm,
):
    current_D_steps, train_generator = 0, True
    model.train()
    device = next(model.parameters()).device
    gan_loss = GANLoss()
    lambda_list = [0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.0483, 0.0932, 0.18]
    beta = [0.0, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56]
    log_base = 5
    lambda_perc = 4.26 / 2.56

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        aux_optimizer.zero_grad()

        q = random.randint(1, 8)
        q_task = random.randint(1, 8)
        q_task = 1
        out_net = model(d, q, q_task)

        lambda_rd = lambda_list[8 - q] * 100
        lambda_gan = beta[8 - q_task]
        ratio = (log_base**((q - 1) / 7) - 1) / (log_base - 1)
        ratio_loss = compute_g_a_ratio_loss(out_net["decisions"], g_a_ratio=ratio)

        out_criterion = criterion(out_net, d)
        x_gen = out_net["x_hat"]
        x_real = d
        D_in = torch.cat([x_real, x_gen], dim=0)
        latents = out_net["latent"]["y_hat"].detach()
        D_out, D_out_logits = model_disc(D_in, latents.repeat(2, 1, 1, 1), q)
        D_out = torch.squeeze(D_out)
        D_out_logits = torch.squeeze(D_out_logits)
        D_real_logits, D_gen_logits = torch.chunk(D_out_logits, 2, dim=0)
        G_loss = gan_loss(D_real_logits, D_gen_logits, mode="generator_loss")
        D_loss = gan_loss(D_real_logits, D_gen_logits, mode="discriminator_loss")

        if train_generator is True:
            optimizer.zero_grad()
            rd_loss = out_criterion["bpp_loss"] * lambda_rd + out_criterion["mse_loss"] / 100
            total_loss = rd_loss + (G_loss + lambda_perc * out_criterion["lpips"]) * lambda_gan + ratio_loss * 10.0
            total_loss.backward()
            aux_loss = model.aux_loss()
            aux_loss.backward()

            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            
            optimizer.step()
            aux_optimizer.step()
            train_generator = False
        else:
            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()
            current_D_steps += 1
            if current_D_steps == 1:
                current_D_steps = 0
                train_generator = True
                continue

        if (i*len(d) % 1000 == 0):
            logging.info(
                f'[{i*len(d)}/{len(train_dataloader.dataset)}] (q={q})| '
                f'Loss: {total_loss.item():.4f} | '
                f'MSE loss: {out_criterion["mse_loss"].item():.5f} | '
                f'LPIPS loss: {out_criterion["lpips"].item():.5f} | '
                f'Bpp loss: {out_criterion["bpp_loss"].item():.4f} | '
                f'G loss: {G_loss.item():.4f} | '
                f'D loss: {D_loss.item():.4f} | '
                f'E Ratio loss ({ratio:.4f}): {ratio_loss.item():.4f} | '
                f"Aux loss: {aux_loss.item():.2f}"
            )


def test_epoch(epoch, test_dataloader, model, model_disc, criterion):
    model.eval()
    device = next(model.parameters()).device
    gan_loss = GANLoss()
    lambda_list = [0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.0483, 0.0932, 0.18]
    # lambda_list = [0.0004, 0.0008, 0.0016, 0.0032, 0.0075, 0.015, 0.03, 0.045]
    # lambda_list = [0.0032, 0.005, 0.0075, 0.013, 0.025, 0.045, 0.065, 0.09]
    beta = [0.0, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56]
    lambda_perc = 4.26 / 2.56

    total_loss = AverageMeter()
    total_bpp_loss = AverageMeter()
    total_mse_loss = AverageMeter()
    total_lpips_loss = AverageMeter()
    total_g_loss = AverageMeter()
    total_d_loss = AverageMeter()
    total_aux_loss = AverageMeter()
    lambda_gan = beta[8 - 1]

    for q in range(1, 9):
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()
        lpips_loss = AverageMeter()
        g_loss = AverageMeter()
        d_loss = AverageMeter()
        aux_loss = AverageMeter()
        ratio = AverageMeter()
        lambda_rd = lambda_list[8 - q] * 100

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d, quality=q, q_task=1)

                out_criterion = criterion(out_net, d)
                x_gen = out_net["x_hat"]
                x_real = d
                D_in = torch.cat([x_real, x_gen], dim=0)
                latents = out_net["latent"]["y_hat"].detach()
                D_out, D_out_logits = model_disc(D_in, latents.repeat(2, 1, 1, 1), q)
                D_out = torch.squeeze(D_out)
                D_out_logits = torch.squeeze(D_out_logits)
                D_real_logits, D_gen_logits = torch.chunk(D_out_logits, 2, dim=0)
                G_loss = gan_loss(D_real_logits, D_gen_logits, mode="generator_loss")
                D_loss = gan_loss(D_real_logits, D_gen_logits, mode="discriminator_loss")

                rd_loss = out_criterion["bpp_loss"] * lambda_rd + out_criterion["mse_loss"] / 100
                egd_loss = rd_loss + (G_loss + lambda_perc * out_criterion["lpips"]) * lambda_gan

                ratio.update(compute_ratio(out_net["decisions"]), d.size(0))
                aux_loss.update(model.aux_loss(), d.size(0))
                bpp_loss.update(out_criterion["bpp_loss"], d.size(0))
                mse_loss.update(out_criterion["mse_loss"], d.size(0))
                lpips_loss.update(out_criterion["lpips"], d.size(0))
                g_loss.update(G_loss, d.size(0))
                d_loss.update(D_loss, d.size(0))
                loss.update(egd_loss, d.size(0))
                

        logging.info(
            f"Test epoch {epoch}: Average losses (q={q}): "
            f"Loss: {loss.avg:.4f} | "
            f"MSE loss: {mse_loss.avg:.5f} | "
            f"LPIPS loss: {lpips_loss.avg:.5f} | "
            f"Bpp loss: {bpp_loss.avg:.4f} | "
            f"G loss: {g_loss.avg:.4f} | "
            f"D loss: {d_loss.avg:.4f} | "
            f"Ratio: {ratio.avg:.4f} | "
            f"Aux loss: {aux_loss.avg:.2f}"
        )

        total_loss.update(loss.avg)
        total_mse_loss.update(mse_loss.avg)
        total_lpips_loss.update(lpips_loss.avg)
        total_bpp_loss.update(bpp_loss.avg)
        total_g_loss.update(g_loss.avg)
        total_d_loss.update(d_loss.avg)
        total_aux_loss.update(aux_loss.avg)

    logging.info(
        f"Test epoch {epoch}: Average losses (total): "
        f"Loss: {total_loss.avg:.4f} | "
        f"MSE loss: {total_mse_loss.avg:.5f} | "
        f"LPIPS loss: {total_lpips_loss.avg:.5f} | "
        f"Bpp loss: {total_bpp_loss.avg:.4f} | "
        f"G loss: {total_g_loss.avg:.4f} | "
        f"D loss: {total_d_loss.avg:.4f} | "
        f"Aux loss: {total_aux_loss.avg:.2f}\n"
    )

    return total_loss.avg


def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    torch.save(state, base_dir+filename)
    if is_best:
        shutil.copyfile(base_dir+filename, base_dir+"checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="mpa_enc",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-i",
        "--task_idx",
        default=0,
        type=int,
        help="Task index (0: MSE, 1: Cls, 2: Seg)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=400,
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
        default=(300,),
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
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux_learning_rate",
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

    train_transforms = transforms.Compose([
        transforms.RandomCrop(args.patch_size, pad_if_needed=True, padding_mode='reflect'), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
        ]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="val", transform=test_transforms)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
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

    net_disc = DiscriminatorHiFiC_Independent()
    net_disc = net_disc.to(device)

    criterion = RateDistortionLoss().to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
        net_disc = CustomDataParallel(net_disc)

    init_weights(net_disc, init_type='normal', init_gain=0.02)

    if args.pretrained:  # load pretrained model
        logging.info("Loading "+str(args.pretrained))
        try:
            pretrained = torch.load(args.pretrained, map_location=device)["state_dict"]
            net.load_state_dict(pretrained, strict=False)
            net.update(force=True)
            if "state_dict_D" in torch.load(args.pretrained, map_location=device).keys():
                net_disc.load_state_dict(torch.load(args.pretrained, map_location=device)["state_dict_D"], strict=False)
                logging.info("Loading pretrained discriminator")
        except KeyError:
            pretrained = torch.load(args.pretrained, map_location=device)
            net.load_state_dict(pretrained, strict=False)

    macs, params = get_model_complexity_info(net.eval(), (3, 256, 256), as_strings=False, print_per_layer_stat=False)
    logging.info("MACs/pixel:"+str(macs/(256**2)))
    logging.info("params:"+str(params))

    optimizer, aux_optimizer = configure_optimizers(net, args)
    optimizer_D = torch.optim.Adam(net_disc.parameters(), lr=args.learning_rate_d)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=args.milestones,
                                                  gamma=args.gamma)
    lr_scheduler_D = optim.lr_scheduler.MultiStepLR(optimizer_D,
                                                  milestones=args.milestones,
                                                  gamma=args.gamma)

    last_epoch = 0
    best_loss = float("inf")
    if args.checkpoint:  # load from previous checkpoint
        logging.info("Loading "+str(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["loss"]
        net.load_state_dict(checkpoint["state_dict"], strict=True)
        net_disc.load_state_dict(checkpoint["state_dict_D"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer_D.load_state_dict(checkpoint["optimizer_D"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        lr_scheduler_D.load_state_dict(checkpoint["lr_scheduler_D"])

    for epoch in range(last_epoch, args.epochs):
        logging.info('======Current epoch %s ======'%epoch)
        logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            net_disc,
            criterion,
            train_dataloader,
            optimizer,
            optimizer_D,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(epoch, test_dataloader, net, net_disc, criterion)
        lr_scheduler.step()
        lr_scheduler_D.step()

        is_best = (loss <= best_loss)
        best_loss = min(loss, best_loss)

        if is_best:
            logging.info(f'Save epoch {epoch} as the best')

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "state_dict_D": net_disc.state_dict(),
                    "loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                    "optimizer_D": optimizer_D.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "lr_scheduler_D": lr_scheduler_D.state_dict(),
                },
                is_best,
                base_dir
            )


if __name__ == "__main__":
    main(sys.argv[1:])
