import os, sys
sys.path.append('./Pytorch-UNet/')
import torch
from torch import optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
import wandb

from datasets import ImagenetDataAugDataset
from wt_utils import wt, create_filters, load_checkpoint, load_weights
from arguments import parse_args
from unet.unet_model import UNet_NTail_128_Mod, UNet_NTail_128_Mod1
from train import train_unet_128_256_perceptual
from logger import Logger

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

    # Initialize wandb
    wandb.init(project=args.project_name)

    # Create transforms
    default_transform = transforms.Compose([
                            transforms.CenterCrop(args.image_size),
                            transforms.Resize(args.image_size),
                            transforms.ToTensor()
                        ])

    # Create train dataset
    train_dataset = dset.ImageFolder(root=args.train_dir, transform=default_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers,
                                               pin_memory=True, drop_last=True)

    # Create validation dataset
    valid_dataset = dset.ImageFolder(root=args.valid_dir, transform=default_transform)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers,
                                               pin_memory=True, drop_last=True)

    # Model creating/loading
    model_128 = UNet_NTail_128_Mod(n_channels=12, n_classes=3, n_tails=12, bilinear=True).to(args.device)
    if args.model_128_weights:
        print('Loading model 128 weights')
        model_128 = load_weights(model_128, args.model_128_weights, args)
    
    model_256 = UNet_NTail_128_Mod(n_channels=48, n_classes=3, n_tails=48, bilinear=True).to(args.device)
    if args.model_256_weights:
        print('Loading model 256 weights')
        model_256 = load_weights(model_256, args.model_256_weights, args)

    optimizer = optim.Adam(list(model_128.parameters()) + list(model_256.parameters()), lr=args.lr)
    # May have to load optimizer when checkpointing
    state_dict = {'itr': args.checkpoint}

    for epoch in range(args.num_epochs):
        train_unet_128_256_perceptual(epoch, state_dict, model_128, model_256, optimizer, train_loader, valid_loader, args, logger)

