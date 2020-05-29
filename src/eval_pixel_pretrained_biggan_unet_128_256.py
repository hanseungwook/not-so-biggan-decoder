import os, sys
import torch
import torchvision.transforms as transforms
import wandb

from arguments_eval import parse_args

import sys
import os
sys.path.insert(0, 'Pytorch-UNet')
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torchvision.utils as vutils
import torchvision
from unet.unet_model import UNet
from torch.nn import functional as F
from logger import Logger
from UNET_utils import load_UNET_checkpoint, load_UNET_weights
from eval_pixel import eval_pretrained_biggan_unet_128_256
from datasets import SampleDataset

if __name__ == "__main__":
    # Set up logger
    logger = Logger()
    
    # Accelerate training with benchmark true
    torch.backends.cudnn.benchmark = True

    # Parse arguments & log
    args = parse_args()
    logger.update_args(args)

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print('WARNING: Output directory already exists and will be overwriting (if not resuming)')

    # Create train dataset
    dataset = SampleDataset(file_path=args.sample_file)
        
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers,
                                              pin_memory=True, drop_last=True)

    model_128 = UNet(n_channels=3, n_classes=3, bilinear=True)
    model_128.to(args.device)
    model_128 = load_UNET_weights(model_128, '128', args)
    
    model_256 = UNet(n_channels=3, n_classes=3, bilinear=True)
    model_256.to(args.device)
    model_256 = load_UNET_weights(model_256, '256', args)

    eval_pretrained_biggan_unet_128_256(model_128, model_256, data_loader, args)
