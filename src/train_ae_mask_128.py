import os, sys
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from vae_models import WT, wt, IWT, iwt, AE_Mask_128
from wt_datasets import CelebaDataset
from trainer import train_ae_mask
from evaluator import eval_ae_mask
from arguments import args_parse
from utils.utils import zero_patches, set_seed, save_plot, create_filters
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
    log_dir = os.path.join(args.root_dir, 'runs/aemask128_{}'.format(args.config))
    try:
        os.mkdir(log_dir)
    except:
        raise Exception('Cannot create log directory')

    writer = SummaryWriter(log_dir=log_dir)

    # Set seed
    set_seed(args.seed)

    dataset_dir = os.path.join(args.root_dir, 'data/celeba_org/celeba64')
    dataset_files = sample(os.listdir(dataset_dir), 10000)
    train_dataset = CelebaDataset(dataset_dir, dataset_files, WT=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=10, shuffle=True)
    sample_dataset = Subset(train_dataset, sample(range(len(train_dataset)), 8))
    sample_loader = DataLoader(sample_dataset, batch_size=8, shuffle=False) 
    
    if args.device >= 0:
        device = 'cuda:{}'.format(args.device)
    else:
        device = 'cpu'
    print('Device: {}'.format(device))
    filters = create_filters(device=device)
    wt_model = WT(wt=wt, num_wt=args.num_iwt)
    wt_model.set_filters(filters)
    wt_model = wt_model.to(device)
    wt_model.set_device(device)

    model = AE_Mask_128(z_dim=args.z_dim)
    model.set_device(device)
    model = model.to(device)
    
    train_losses = []
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    criterion = None
    if args.loss == 'l1':
        criterion = torch.nn.L1Loss()
    elif args.loss == 'l2':
        criterion = torch.nn.MSELoss()
    elif args.loss == 'bce':
        criterion = torch.nn.BCELoss()

    img_output_dir = os.path.join(args.root_dir, 'wtvae_results/image_samples/aemask128_{}'.format(args.config))
    model_dir = os.path.join(args.root_dir, 'wtvae_results/models/aemask128_{}/'.format(args.config))

    try:
        os.mkdir(img_output_dir)
        os.mkdir(model_dir)
    except:
        LOGGER.error('Could not make model & img output directories')
        raise Exception('Could not make model & img output directories')
    
    for epoch in range(1, args.epochs + 1):
        train_ae_mask(epoch, wt_model, model, criterion, optimizer, train_loader, train_losses, args, writer)
        eval_ae_mask(epoch, wt_model, model, sample_loader, args, img_output_dir, model_dir, writer)

    
    # Save train losses and plot
    np.save(model_dir+'/train_losses.npy', train_losses)
    
    LOGGER.info('AE Model parameters: {}'.format(sum(x.numel() for x in model.parameters())))

    
    
    
