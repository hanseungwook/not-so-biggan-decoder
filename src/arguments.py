import argparse

def args_parse():
    """Parses command line arguments for model training"""

    # Arguments shared by WTVAE & IWTVAE
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
    parser.add_argument('--seed', type=int, default=2020,
                        help='Random seed')
    parser.add_argument('--device', type=int, default=-1,
                        help='GPU device # (-1 if CPU)')
    parser.add_argument('--loss', type=str, default='l1',
                        help='Type of loss/criterino to use for loss function')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to checkpoint (saved model & optimizer state) to load and continue training from')
    parser.add_argument('--checkpoint_epoch', type=int, default=1,
                        help='Checkpoint epoch to continue training from')


    # Arguments for controlling KL weight anneal
    parser.add_argument('--kl_start', type=float, default=1.0,
                        help='Starting KL weight')
    parser.add_argument('--kl_warmup', type=int, default=2,
                        help='Number of annealing epochs (weight increases from kl_start to 1.0 linearly in the first warm_up epochs)')    
    parser.add_argument('--kl_weight', type=float, default=1.0,
                        help='Saving current KL weight as arg') 

    parser.add_argument('--grad_clip', type=float, default=0.0,
                        help='Max norm to clip the gradient at (>0 in order to do any gradient clipping)') 

    # Arguments exclusively for IWTVAE
    parser.add_argument('--upsampling', type=str, default='linear',
                        help='Upsampling layer for IWTVAE: linear, conv1d, conv2d') 
    parser.add_argument('--reuse', action='store_true', default=False,
                        help='Whether to re-use upsampling layer or not')
    parser.add_argument('--zero', action='store_true', default=False,
                        help='Whether to zero out patches other than the first or not')
    parser.add_argument('--num_upsampling', type=int, default=2,
                        help='Number of upsampling layers')
    parser.add_argument('--wt_model', type=str, default='',
                        help='Saved model state for wtvae (for inference during iwtvae training)')
    parser.add_argument('--iwt_model', type=str, default='',
                        help='Saved model state for iwtvae (for training wtvae with iwtvae frozen for full pipeline)')
    parser.add_argument('--mask', action='store_true', default=False,
                        help='Whether to learn z as mask or not')  
    parser.add_argument('--bottleneck_dim', type=int, default=0,
                        help='Bottleneck dim for Y bottleneck (>0 to use this model)')    
    parser.add_argument('--freeze_iwt', action='store_true', default=False,
                        help='Whether to train model with frozen IWT')       
    parser.add_argument('--num_iwt', type=int, default=2,
                        help='Number of times to apply deterministic IWT')
    parser.add_argument('--img_loss_epoch', type=int, default=0,
                        help='Epoch # to introduce l2 img reconstruction loss on top of WT-space l1 loss')                 

    # Arguments for FullVAE
    parser.add_argument('--z_dim_wt', type=int, default=100,
                        help='Z dimension for WTVAE model')
    parser.add_argument('--z_dim_iwt', type=int, default=100,
                        help='Z dimension for IWTVAE model')

    args = parser.parse_args()


    return args
