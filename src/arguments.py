import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='UNet training for 128 level masks on whole dataset')

    # Data arguments
    parser.add_argument('--train_dir', type=str, 
                        help='Path for train data')
    parser.add_argument('--valid_dir', type=str, 
                        help='Path for validation data')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size for train dataset')
    parser.add_argument('--image_size', type=int, default=256, 
                        help='Image size for train dataset')
    parser.add_argument('--workers', type=int, default=4, 
                        help='Number of workers for dataloader')                    
    parser.add_argument('--mask_dim', type=int, default=64,
                        help='Dimension of mask trying to reconstruct (32 / 64 / 128')

    # Model arguments
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')

    # Train arguments
    parser.add_argument('--num_epochs', type=int, default=300, 
                        help='Number of epochs to train')
    parser.add_argument('--output_dir', type=str, 
                        help='Output directory to create and save files')

    parser.add_argument('--project_name', type=str, default='unet_full_imagenet_128',
                        help='Project name for wandb')
    parser.add_argument('--save_every', type=int, default=500,
                        help='Save every X iterations')
    parser.add_argument('--valid_every', type=int, default=1000,
                        help='Evaluate validation dataset every X iterations')    
    parser.add_argument('--log_every', type=int, default=50,
                        help='Log train metrics every X iterations')   

    # Resume
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training? (default: %(default)s)')
    parser.add_argument('--checkpoint', type=int, default=0,
                        help='Resume iteration X (default: %(default)s)')

    # Loading weights for eval
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='Path to load weights from (for evaluation) (default: %(default)s)')              

    # Model weights for 128 (when training 256)
    parser.add_argument('--model_128_weights', type=str, default='',
                        help='Path to 128 model weights(default: %(default)s)')

    
    args = parser.parse_args()

    # Use GPU, if available
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return args