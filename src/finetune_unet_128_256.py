import os, sys
sys.path.append('./Pytorch-UNet/')
import torch
from torch import optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
import wandb
import gc

from datasets import parse_dataset_args, create_dataset
from wt_utils import wt, create_filters, load_checkpoint, load_weights, load_all_weights
from arguments import parse_args
from unet.unet_model import UNet_NTail_128_Mod
from train import train_unet_128_256
from losses import DecoderLoss
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

    # Parsing dataset arguments
    ds_name, classes = parse_dataset_args(args.dataset)

    # Create train dataset
    train_dataset = create_dataset(ds_name, args.train_dir, transform=default_transform, classes=classes[0] if classes else None)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers,
                                               pin_memory=True, drop_last=True)

    # Create validation dataset
    valid_dataset = create_dataset(ds_name, args.valid_dir, transform=default_transform, classes=classes[1] if classes else None)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers,
                                               pin_memory=True, drop_last=True)

    print('Loading UNet 128 and 256 weights')
    model_128 = UNet_NTail_128_Mod(n_channels=12, n_classes=3, n_tails=12, bilinear=True).to(args.device)
    model_256 = UNet_NTail_128_Mod(n_channels=48, n_classes=3, n_tails=48, bilinear=True).to(args.device)

    # Optimizer
    optimizer = optim.Adam(list(model_128.parameters()) + list(model_256.parameters()), lr=args.lr)

    # State dict
    state_dict = {'itr': 0}

    # If continuing finetuning/training, load these weights
    if args.resume:
        state_dict['itr'] = args.checkpoint
        model_128, model_256, optimizer = load_all_weights(model_128, model_256, optimizer, args.checkpoint_path, args)
        print('Loaded models and optimizer from iteration {}'.format(args.checkpoint))

    # Else, loading original decoders only trained on MSE
    else: 
        model_128 = load_weights(model_128, args.model_128_weights, args)
        model_256 = load_weights(model_256, args.model_256_weights, args)

    # Decoder loss = perceptual loss (VGG-19)
    loss = DecoderLoss(model_path=args.vgg_model_path, feature_idx=args.feature_idx, bn=args.bn, loss_criterion=args.loss_criterion, use_input_norm=args.use_input_norm, use_wt=args.use_wt, device=args.device)

    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()

    for epoch in range(args.num_epochs):
        train_unet_128_256(epoch, state_dict, model_128, model_256, optimizer, train_loader, valid_loader, args, logger, loss)

