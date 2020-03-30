import os, sys
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from vae_models import WTVAE_64, IWTVAE_512_Mask, WT, wt, IWT, iwt
from wt_datasets import CelebaDataset
from trainer import train_iwtvae
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

    # Setting up tensorboard writer
    log_dir = os.path.join(args.root_dir, 'runs/{}'.format(args.config))
    try:
        os.mkdir(log_dir)
    except:
        raise Exception('Cannot create log directory')

    writer = SummaryWriter(log_dir=log_dir)

    # Set seed
    set_seed(args.seed)

    dataset_dir = os.path.join(args.root_dir, 'data/celebaHQ512')
    dataset_files = sample(os.listdir(dataset_dir), 10000)
    train_dataset = CelebaDataset(dataset_dir, dataset_files, WT=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=10, shuffle=True)
    sample_dataset = Subset(train_dataset, sample(range(len(train_dataset)), 8))
    sample_loader = DataLoader(sample_dataset, batch_size=8, shuffle=False) 
    
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        devices = ['cuda:0', 'cuda:1']
    else: 
        devices = ['cpu', 'cpu']

    inv_filters = create_inv_filters(device=devices[0])
    filters = create_filters(device=devices[0])

    wt_model = WT(wt=wt, num_wt=args.num_iwt)
    wt_model.set_filters(filters)
    wt_model = wt_model.to(devices[0])
    wt_model.set_device(devices[0])

    iwt_model = IWTVAE_512_Mask(z_dim=args.z_dim, num_iwt=args.num_iwt)
    iwt_model.set_filters(inv_filters)
    iwt_model.set_device(devices[0])
    iwt_model = iwt_model.to(devices[0])

    iwt_fn = IWT(iwt=iwt, num_iwt=args.num_iwt)
    iwt_fn.set_filters(inv_filters)
    
    train_losses = []
    optimizer = optim.Adam(iwt_model.parameters(), lr=args.lr)

    img_output_dir = os.path.join(args.root_dir, 'wtvae_results/image_samples/iwtvae512_{}'.format(args.config))
    model_dir = os.path.join(args.root_dir, 'wtvae_results/models/iwtvae512_{}/'.format(args.config))

    try:
        os.mkdir(img_output_dir)
        os.mkdir(model_dir)
    except:
        LOGGER.error('Could not make model & img output directories')
        raise Exception('Could not make model & img output directories')
    
    for epoch in range(1, args.epochs + 1):
        train_iwtvae(epoch, wt_model, iwt_model, optimizer, train_loader, train_losses, args, writer)
        
        with torch.no_grad():
            iwt_model.eval()
            
            for data in sample_loader:
                data = data.to(devices[0])
                
                Y = wt_model(data)
                # Y[:, :, :128, :128] += torch.randn(Y[:, :, :128, :128].shape, device=devices[0])
                save_image(Y.cpu(), img_output_dir + '/sample_y_before_zero{}.png'.format(epoch))
                Y_full = Y.clone()
                if args.zero:
                    Y = zero_patches(Y, num_wt=args.num_iwt)
                Y = Y.to(devices[0])

                z_sample = torch.randn(data.shape[0],args.z_dim).to(devices[0])
    
                mu, var, m1_idx, m2_idx = iwt_model.encode(Y_full - Y)
                # x_hat = iwt_model.decode(Y, mu, m1_idx, m2_idx)
                # x_sample = iwt_model.decode(Y, z_sample, m1_idx, m2_idx)
                x_wt_hat = iwt_model.decode(Y, mu, m1_idx, m2_idx)
                x_wt_sample = iwt_model.decode(Y, z_sample, m1_idx, m2_idx)

                # x_wt_hat = postprocess_low_freq(x_wt_hat)
                # x_wt_sample = postprocess_low_freq(x_wt_hat)
                x_hat = iwt_fn(x_wt_hat)
                x_sample = iwt_fn(x_wt_sample)

                # x_hat_wt = wt_model(x_hat)
                x_hat_wt = x_wt_hat
                unmasked_x = x_hat_wt[:, :, :128, :128]
                masked_x = x_hat_wt[:, :, 128:, :]
                writer.add_histogram('Unmasked_values', unmasked_x.reshape(-1).cpu(), epoch)
                writer.add_histogram('Masked_values', masked_x.reshape(-1).cpu(), epoch)
                writer.flush()
                
                save_image(x_hat.cpu(), img_output_dir + '/sample_recon{}.png'.format(epoch))
                save_image(x_hat_wt.cpu(), img_output_dir + '/sample_recon_wt{}.png'.format(epoch))
                save_image(x_sample.cpu(), img_output_dir + '/sample_z{}.png'.format(epoch))
                save_image(Y.cpu(), img_output_dir + '/sample_y{}.png'.format(epoch))
                save_image(data.cpu(), img_output_dir + '/sample{}.png'.format(epoch))
    
        torch.save(iwt_model.state_dict(), model_dir + '/iwtvae_epoch{}.pth'.format(epoch))
    
    # Save train losses and plot
    np.save(model_dir+'/train_losses.npy', train_losses)
    
    LOGGER.info('IWT Model parameters: {}'.format(sum(x.numel() for x in iwt_model.parameters())))

    
    
    
