import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='UNET for Imagenet')

    # Data arguments
    parser.add_argument('--train_dir', type=str, 
                        help='Path for train data')
    parser.add_argument('--valid_dir', type=str, 
                        help='Path for validation data')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size for train dataset')
    parser.add_argument('--workers', type=int, default=4, 
                        help='Number of workers for dataloader')     
    
    parser.add_argument("--low_resolution", help="downsampled resolution", default=128, type=int)
    parser.add_argument('--image_size', type=int, default=256, 
                        help='Image size for train dataset')

    # Model arguments
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='Learning rate')

    # Train arguments
    parser.add_argument('--num_epochs', type=int, default=100, 
                        help='Number of epochs to train')
    parser.add_argument('--output_dir', type=str, 
                        help='Output directory to create and save files')

    parser.add_argument('--project', type=str, default='UNET',
                        help='Project name for wandb')

    parser.add_argument('--save_every', type=int, default=500,
                        help='Save every X iterations')
    parser.add_argument('--valid_every', type=int, default=5000,
                        help='Evaluate validation dataset every X iterations')    
    # Resume
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training? (default: %(default)s)')
    parser.add_argument('--checkpoint', type=int, default=0,
                        help='Resume iteration X (default: %(default)s)')

    
    args = parser.parse_args()

    # Use GPU, if available
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return args
