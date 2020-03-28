import os, sys
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from vae_models import WTVAE_64_1
from wt_datasets import CelebaDataset
from trainer import train_wtvae
from evaluator import eval_wtvae
from arguments import args_parse
from utils.utils import set_seed, save_plot, zero_pad, create_filters
import logging
from random import sample


if __name__ == "__main__":
    # Accelerate training since fixed input sizes
    torch.backends.cudnn.benchmark = True 

    # Setting up logger
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(message)s')
    LOGGER = logging.getLogger(__name__)

    args = args_parse()

    # Set seed
    set_seed(args.seed)

    # Create training and sample dataset (to test out model and save images for)
    dataset_dir = os.path.join(args.root_dir, 'data/celeba256')
    dataset_files = sample(os.listdir(dataset_dir), 10000)
    train_dataset = CelebaDataset(dataset_dir, dataset_files, WT=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=10, shuffle=True)
    sample_dataset = Subset(train_dataset, sample(range(len(train_dataset)), 8))
    sample_loader = DataLoader(sample_dataset, batch_size=8, shuffle=False) 
    
    if torch.cuda.is_available() and device >= 0:
        device = 'cuda:{}'.format(args.device)
    else: 
        device = 'cpu'

    # Setting up WT & IWT filters
    filters = create_filters(device=device, wt_fn='bior2.2')

    # Create model, set filters for WT (calculating loss), and set device
    wt_model = WTVAE_64_1(z_dim=args.z_dim, num_wt=args.num_wt)
    wt_model = wt_model.to(device)
    wt_model.set_filters(filters)
    wt_model.set_device(device)
    
    train_losses = []
    optimizer = optim.Adam(wt_model.parameters(), lr=args.lr)

    # Create output and log directories
    img_output_dir = os.path.join(args.root_dir, 'wtvae_results/image_samples/wtvae64_{}'.format(args.config))
    model_dir = os.path.join(args.root_dir, 'wtvae_results/models/wtvae64_{}/'.format(args.config))
    log_dir = os.path.join(args.root_dir, 'runs/{}'.format(args.config))    
    
    try:
        os.mkdir(img_output_dir)
        os.mkdir(model_dir)
        os.mkdir(log_dir)
    except:
        LOGGER.error('Could not make model / img / log output directories')
        raise Exception('Could not make model / img / log output directories')

    # Setting up tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.root_dir, 'runs'))

    # Setting current kl weight to start weight
    args.kl_weight = args.kl_start

    for epoch in range(1, args.epochs + 1):
        train_wtvae(epoch, wt_model, optimizer, train_loader, train_losses, args, writer)
        eval_wtvae(epoch, wt_model, sample_loader, args, img_output_dir, model_dir)
    
    # Save train losses and plot
    np.save(model_dir+'/train_losses.npy', train_losses)
    save_plot(train_losses, img_output_dir + '/train_loss.png')
    
    writer.close()
    LOGGER.info('WT Model parameters: {}'.format(sum(x.numel() for x in wt_model.parameters())))

    
    
    