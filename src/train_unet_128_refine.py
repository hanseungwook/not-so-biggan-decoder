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
from train import train_unet128_refine
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

    # Model and optimizer
    print('Loading model 128 weights')
    model_128 = UNet_NTail_128_Mod(n_channels=12, n_classes=3, n_tails=12, bilinear=True).to(args.device)
    model_128 = load_weights(model_128, args.model_128_weights, args)

    model = UNet_NTail_128_Mod1(n_channels=36, n_classes=3, n_tails=12, bilinear=True).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    state_dict = {'itr': 0}

    if args.resume:
        print('Loading weights & resuming from iteration {}'.format(args.checkpoint))
        model, optimizer, logger = load_checkpoint(model, optimizer, '128', args)
        state_dict['itr'] = args.checkpoint

    for epoch in range(args.num_epochs):
        train_unet128_refine(epoch, state_dict, model_128, model, optimizer, train_loader, valid_loader, args, logger)