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
from unet.unet_model import UNet_NTail_128_Mod
from eval import eval_unet256
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
        os.mkdir(args.output_dir + '/train/')
        os.mkdir(args.output_dir + '/valid/')
    else:
        print('WARNING: Output directory already exists and will be overwriting (if not resuming)')
    
    # Create filters for dataloader
    filters_cpu = create_filters(device='cpu')

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

    # Model and optimizer
    model = UNet_NTail_128_Mod(n_channels=48, n_classes=3, n_tails=48, bilinear=True).to(args.device)
    
    # Load weights
    print('Loading weights')
    model = load_weights(model, args.model_256_weights, args)

    eval_unet256(model, train_loader, 'train', args)
    eval_unet256(model, valid_loader, 'valid', args)

