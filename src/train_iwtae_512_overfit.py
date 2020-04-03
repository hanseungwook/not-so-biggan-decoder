import os, sys
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from vae_models import WT, wt, IWT, iwt, IWTAE_512_Mask_2
from wt_datasets import CelebaDataset
from trainer import train_iwtae_iwtmask
from evaluator import eval_iwtae_iwtmask
from arguments import args_parse
from utils.utils import zero_patches, set_seed, save_plot, create_inv_filters, create_filters
import matplotlib.pyplot as plt
import logging
import pywt
from random import sample


if __name__ == "__main__":
    # Accelerate training since fixed input sizes
    torch.backends.cudnn.benchmark = True 

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(message)s')
    LOGGER = logging.getLogger(__name__)

    args = args_parse()

    # Set seed
    set_seed(args.seed)

    # Create dataset
    dataset_dir = os.path.join(args.root_dir, 'data/celebaHQ512')
    dataset_files = sample(os.listdir(dataset_dir), 10000)
    train_dataset = CelebaDataset(dataset_dir, dataset_files, WT=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=10, shuffle=True)
    sample_dataset = Subset(train_dataset, sample(range(len(train_dataset)), 4))
    sample_loader = DataLoader(sample_dataset, batch_size=4, shuffle=False) 
    
    if args.device >= 0:
        device = 'cuda:{}'.format(args.device)
        print('Device {}'.format(device))
    else: 
        device = 'cpu'

    inv_filters = create_inv_filters(device=device)
    filters = create_filters(device=device)

    wt_model = WT(wt=wt, num_wt=args.num_iwt)
    wt_model.set_filters(filters)
    wt_model = wt_model.to(device)
    wt_model.set_device(device)

    iwt_model = IWTAE_512_Mask_2(z_dim=args.z_dim, num_iwt=args.num_iwt)
    iwt_model.set_filters(inv_filters)
    iwt_model.set_device(device)
    iwt_model = iwt_model.to(device)

    iwt_fn = IWT(iwt=iwt, num_iwt=args.num_iwt)
    iwt_fn.set_filters(inv_filters)
    
    train_losses = []
    optimizer = optim.Adam(iwt_model.parameters(), lr=args.lr)

    img_output_dir = os.path.join(args.root_dir, 'wtvae_results/image_samples/iwtae512_overfit_{}'.format(args.config))
    model_dir = os.path.join(args.root_dir, 'wtvae_results/models/iwtae512_overfit_{}/'.format(args.config))
    log_dir = os.path.join(args.root_dir, 'runs/iwtae512_overfit_{}'.format(args.config))

    try:
        os.mkdir(img_output_dir)
        os.mkdir(model_dir)
        os.mkdir(log_dir)
    except:
        LOGGER.error('Could not make log / model / img output directories')
        raise Exception('Could not make log / model / img output directories')

    # Set up tensorboard logger
    writer = SummaryWriter(log_dir=log_dir)
    
    # Annealing of KL weight over each epoch
    args.kl_weight = args.kl_start
    anneal_rate = (1.0 - args.kl_start) / (args.kl_warmup)

    for epoch in range(1, args.epochs + 1):
        args.kl_weight = min(1.0, args.kl_weight + anneal_rate)
        train_iwtae_iwtmask(epoch, wt_model, iwt_model, optimizer, iwt_fn, sample_loader, train_losses, args, writer)
        eval_iwtae_iwtmask(epoch, wt_model, iwt_model, optimizer, iwt_fn, sample_loader, args, img_output_dir, model_dir, writer, not args.nosave)
    
    # Save train losses and plot
    np.save(model_dir+'/train_losses.npy', train_losses)
    
    LOGGER.info('IWT Model parameters: {}'.format(sum(x.numel() for x in iwt_model.parameters())))