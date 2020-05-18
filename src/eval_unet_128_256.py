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
from eval import eval_unet_128_256
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
    
    # Image augmentation transforms + dataset
    # transform = [transforms.ColorJitter(0.5, 0.5, 0.5, 0),
    #             transforms.RandomAffine(180),
    #             transforms.RandomErasing(p=1, value=1)]

    # train_dataset = ImagenetDataAugDataset(root_dir=args.train_dir, num_wt=3, mask_dim=args.mask_dim, wt=wt, 
    #                                        filters=filters_cpu, default_transform=default_transform,
    #                                        transform=transform, p=0.1)

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

    # Model
    print('Loading UNet 128 and 256 weights')
    model_128 = UNet_NTail_128_Mod(n_channels=12, n_classes=3, n_tails=12, bilinear=True).to(args.device)
    model_128 = load_weights(model_128, args.model_128_weights, args)
    
    model_256 = UNet_NTail_128_Mod(n_channels=48, n_classes=3, n_tails=48, bilinear=True).to(args.device)
    model_256 = load_weights(model_256, args.model_256_weights, args)

    eval_unet_128_256(model_128, model_256, train_loader, 'train', args)
    eval_unet_128_256(model_128, model_256, valid_loader, 'valid', args)

