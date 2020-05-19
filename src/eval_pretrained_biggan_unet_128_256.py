import os, sys
sys.path.append('./Pytorch-UNet/')
import torch
from torch import optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
import wandb

from datasets import SampleDataset
from wt_utils import wt, create_filters, load_checkpoint, load_weights
from arguments import parse_args
from unet.unet_model import UNet_NTail_128_Mod
from eval import eval_pretrained_biggan_unet_128_256
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
    
    # Create filters for dataloader
    filters_cpu = create_filters(device='cpu')

    # Create transforms
    default_transform = transforms.Compose([
                            transforms.CenterCrop(args.image_size),
                            transforms.Resize(args.image_size),
                            transforms.ToTensor()
                        ])

    # Create train dataset
    dataset = SampleDataset(file_path=args.sample_file)
        
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers,
                                               pin_memory=True, drop_last=True)

    print('Loading UNet 128 and 256 weights')
    model_128 = UNet_NTail_128_Mod(n_channels=12, n_classes=3, n_tails=12, bilinear=True).to(args.device)
    model_128 = load_weights(model_128, args.model_128_weights, args)
    
    model_256 = UNet_NTail_128_Mod(n_channels=48, n_classes=3, n_tails=48, bilinear=True).to(args.device)
    model_256 = load_weights(model_256, args.model_256_weights, args)

    eval_pretrained_biggan_unet_128_256(model_128, model_256, data_loader, args)

