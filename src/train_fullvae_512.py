"""
Script for training full pipeline with WTVAE 128 (1) + IWTAE 512 (frozen)
"""
import os, sys
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import numpy as np
from vae_models import WTVAE_128_1, IWTAE_512_Mask_2, Full_WTVAE128_IWTAE512
from wt_datasets import CelebaDatasetPair
from trainer import train_full_wtvae128_iwtae512
from evaluator import eval_full_wtvae128_iwtae512
from arguments import args_parse
from utils.utils import set_seed, save_plot, zero_pad, create_filters, create_inv_filters
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

    # Setting up tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.root_dir, 'runs'))

    # Set seed
    set_seed(args.seed)

    dataset_dir_128 = os.path.join(args.root_dir, 'data/celeba128')
    dataset_dir_512 = os.path.join(args.root_dir, 'data/celebaHQ512')
    dataset_files = sample(os.listdir(dataset_dir_512), 10000)
    train_dataset = CelebaDatasetPair(dataset_dir_128, dataset_dir_512, dataset_files, WT=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=10, shuffle=True)
    sample_dataset = Subset(train_dataset, sample(range(len(train_dataset)), 8))
    sample_loader = DataLoader(sample_dataset, batch_size=8, shuffle=False) 
    
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        devices = ['cuda:0', 'cuda:1']
    else: 
        devices = ['cpu', 'cpu']

    # Setting up WT & IWT filters
    filters = create_filters(device=devices[0])
    inv_filters = create_inv_filters(device=devices[1])

    wt_model = WTVAE_128_1(z_dim=args.z_dim, num_wt=args.num_iwt)
    wt_model.set_filters(filters)
    iwt_model = IWTAE_512_Mask_2(z_dim=args.z_dim, num_iwt=args.num_iwt)
    iwt_model.set_filters(inv_filters)
    
    # If given saved IWT model, load and freeze model
    if args.iwt_model:
        checkpoint = torch.load(args.iwt_model, map_location=devices[1])
        iwt_model.load_state_dict(checkpoint['model_state_dict'])
        for param in iwt_model.parameters():
            param.requires_grad = False
            
    full_model = Full_WTVAE128_IWTAE512(wt_model=wt_model, iwt_model=iwt_model, devices=devices)
    
    train_losses = []

    if args.iwt_model:
        optimizer = optim.Adam(wt_model.parameters(), lr=args.lr)
    elif args.wt_model:
        optimizer = optim.Adam(iwt_model.parameters(), lr=args.lr)
    else: 
        optimizer = optim.Adam(list(wt_model.parameters()) + list(iwt_model.parameters()), lr=args.lr)

    img_output_dir = os.path.join(args.root_dir, 'wtvae_results/image_samples/fullvae512_{}'.format(args.config))
    model_dir = os.path.join(args.root_dir, 'wtvae_results/models/fullvae512_{}/'.format(args.config))

    try:
        os.mkdir(img_output_dir)
        os.mkdir(model_dir)
    except:
        LOGGER.error('Could not make model & img output directories')
        raise Exception('Could not make model & img output directories')
    
    for epoch in range(1, args.epochs + 1):
        train_full_wtvae128_iwtae512(epoch, full_model, optimizer, train_loader, train_losses, args, writer)
        eval_full_wtvae128_iwtae512(epoch, full_model, optimizer, sample_loader, args, img_output_dir, model_dir, writer, args.nosave)

    # Save train losses and plot
    np.save(model_dir+'/train_losses.npy', train_losses)
    save_plot(train_losses, img_output_dir + '/train_loss.png')
    
    LOGGER.info('Full Model parameters: {}'.format(sum(x.numel() for x in full_model.wt_model.parameters()) + sum(x.numel() for x in full_model.iwt_model.parameters())))

    
    
    