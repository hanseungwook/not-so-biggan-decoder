import os, sys
import torch
import torchvision.transforms as transforms
import wandb

from datasets import ImagenetDataset

from arguments_256_pixel import parse_args
from train_256_pixel import train_256

import sys
import os
sys.path.insert(0, 'Pytorch-UNet')
sys.path.insert(0, 'utils')
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torchvision.utils as vutils
import torchvision
from unet.unet_model import UNet
from torch.nn import functional as F
from logger import Logger
from UNET_utils import load_UNET_checkpoint


if __name__ == "__main__":
    # Set up logger
    logger = Logger()

    # Parse arguments & log
    args = parse_args()
    logger.update_args(args)
    
    # Accelerate training with benchmark true
    torch.backends.cudnn.benchmark = True

    # Create output directory
    try:
        os.mkdir(args.output_dir)
    except:
        print('Output directory already exists')

    # Initialize wandb
    wandb.init(project=args.project)
    
    # Create datasets
    default_transform = transforms.Compose([
                            transforms.CenterCrop(args.image_size),
                            transforms.Resize(args.image_size),
                            transforms.ToTensor()
                        ])

    # Create training dataset
    
    train_dataset = dset.ImageFolder(root=args.train_dir, transform=default_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers,
                                               pin_memory=True, drop_last=True)

    # Create validation dataset
    
    valid_data = dset.ImageFolder(root=args.valid_dir, transform=default_transform)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64,
                                               shuffle=True, num_workers=args.workers,
                                               pin_memory=True, drop_last=True)
    
    model = UNet(n_channels=3, n_classes=3, bilinear=True)
    model.to(args.device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=0)
    
    state_dict = {'itr': 0}
    
    if args.resume:
        print('Loading weights & resuming from iteration {}'.format(args.checkpoint))
        model, optimizer, logger = load_UNET_checkpoint(model, optimizer, '256', args)
        state_dict['itr'] = args.checkpoint
    
    for epoch in range(args.num_epochs):
        train_256(epoch, state_dict, model, optimizer, train_loader, valid_loader, args, logger)