import argparse

def args_parse():
    """Parses command line arguments for model training"""
    parser = argparse.ArgumentParser(description='Arguments for training Wavelet VAE')
    parser.add_argument('--epochs', type=int, default=100,
                        help='# of epochs for training')
    parser.add_argument('--root_dir', type=str, default='',
                        help='Root directory to load or save models/data')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate of optimizer')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log interval of training / evaluation progress')               
    parser.add_argument('--config', type=str, default='',
                        help='Configuration for model -- to be used for output directory')
    parser.add_argument('--unflatten', type=int, default=0,
                        help='Which unflatten to use for WT VAE decoder (0/1)')
    parser.add_argument('--num_wt', type=int, default=2,
                        help='How many WT in WT VAE')
    parser.add_argument('--z_dim', type=int, default=100,
                        help='Z dimension (whether in WTVAE or IWTVAE)')
    parser.add_argument('--upsampling', type=str, default='linear',
                        help='Upsampling layer for IWTVAE: linear, conv1d, conv2d') 
    parser.add_argument('--reuse', action='store_true', default=False,
                        help='Whether to re-use upsampling layer or not')
    parser.add_argument('--num_upsampling', type=int, default=2,
                        help='Number of upsampling layers')
    parser.add_argument('--wtvae_model', type=str, default='',
                        help='Saved model state for wtvae (for inference during iwtvae training)')

    args = parser.parse_args()


    return args
