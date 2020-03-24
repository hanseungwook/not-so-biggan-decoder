import os, sys
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import numpy as np
from vae_models import WTVAE_128_1, IWTVAE_512_Mask, FullVAE_512
from wt_datasets import CelebaDatasetPair
from trainer import train_fullvae
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

    # Set seed
    set_seed(args.seed)

    # Create training and sample dataset (to test out model and save images for)
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


    wt = WT(num_wt=2)
    wt_model = WTVAE_128_1(z_dim=args.z_dim_wt, num_wt=args.num_iwt)
    wt_model.set_filters(filters)
    iwt_model = IWTVAE_512_Mask(z_dim=args.z_dim_iwt, num_iwt=args.num_iwt)
    iwt_model.set_filters(inv_filters)
    
    # Given saved model, load and freeze model
    if args.iwt_model and args.wt_model:
        iwt_model.load_state_dict(torch.load(args.iwt_model))
        wt_model.load_state_dict(torch.load(args.wt_model))
            
    full_model = FullVAE_512(wt_model=wt_model, iwt_model=iwt_model, devices=devices)

    img_output_dir = os.path.join(args.root_dir, 'wtvae_results/image_samples/fullvae128512_{}'.format(args.config))
    model_dir = os.path.join(args.root_dir, 'wtvae_results/models/fullvae128512_{}/'.format(args.config))

    try:
        os.mkdir(img_output_dir)
        os.mkdir(model_dir)
    except:
        LOGGER.error('Could not make model & img output directories')
        raise Exception('Could not make model & img output directories')
            
    with torch.no_grad():
        full_model.eval()
        full_model.wt_model.eval()
        full_model.iwt_model.eval()
        
        for data in sample_loader:
            data128 = data[0]
            data512 = data[1]
            z, mu_wt, logvar_wt = full_model.wt_model.encode(data128.to(devices[0]))

            # Creating z sample for WT model by adding Gaussian noise ~ N(0,1)
            z_sample1 = torch.randn(z.shape).to(devices[0])
            z_sample3 = z + torch.randn(z.shape).to(devices[0])

            y = full_model.wt_model.decode(z)
            y_sample = full_model.wt_model.decode(z_sample1)
            y_sample_gaussian = full_model.wt_model.decode(z_sample3)

            y_padded = zero_pad(y, target_dim=512, device=devices[1])
            y_sample_padded = zero_pad(y_sample, target_dim=512, device=devices[1])
            y_sample_padded_gaussian = zero_pad(y_sample_gaussian, target_dim=512, device=devices[1])

            y_padded_iwt = y_padded.clone().detach()
            for i in range(2):
                y_padded_iwt = full_model.iwt_model.iwt(y_padded_iwt)
            
            mu, var, m1_idx, m2_idx = full_model.iwt_model.encode(data512.to(devices[1]), y_padded)
            mu_iwt, var_iwt, m1_idx_iwt, m2_idx_iwt = full_model.iwt_model.encode(y_padded_iwt, y_padded)
            z_sample2 = torch.randn(mu.shape).to(devices[1])

            x_hat = iwt_model.decode(y_padded, mu, m1_idx, m2_idx)
            x_sample = iwt_model.decode(y_padded, z_sample2, m1_idx, m2_idx)

            x_hat_iwt = iwt_model.decode(y_padded, mu_iwt, m1_idx_iwt, m2_idx_iwt)
            x_sample_iwt = iwt_model.decode(y_padded, z_sample2, m1_idx_iwt, m2_idx_iwt)
            
            x_sample_y_sample = iwt_model.decode(y_sample_padded, z_sample2, m1_idx, m2_idx)
            x_sample_y_sample_gaussian = iwt_model.decode(y_sample_padded_gaussian, z_sample2, m1_idx, m2_idx)
            x_y_sample_gaussian = iwt_model.decode(y_sample_padded_gaussian, mu, m1_idx, m2_idx)

            x_sample_y_sample_iwt = iwt_model.decode(y_sample_padded, z_sample2, m1_idx_iwt, m2_idx_iwt)
            x_sample_y_sample_gaussian_iwt = iwt_model.decode(y_sample_padded_gaussian, z_sample2, m1_idx_iwt, m2_idx_iwt)
            x_y_sample_gaussian_iwt = iwt_model.decode(y_sample_padded_gaussian, mu_iwt, m1_idx_iwt, m2_idx_iwt)

            x_hat_2wt = wt(x_hat)

            save_image(x_hat_2wt.cpu(), img_output_dir + '/sample_recon_x_2wt.png')

            save_image(x_hat.cpu(), img_output_dir + '/sample_recon_x.png')
            save_image(x_sample.cpu(), img_output_dir + '/sample_z.png')
            save_image(x_sample_y_sample.cpu(), img_output_dir + '/sample_z_both.png')
            save_image(x_sample_y_sample_gaussian.cpu(), img_output_dir + '/sample_z_both_gaussian.png')
            save_image(x_y_sample_gaussian.cpu(), img_output_dir + '/sample_z_wt_gaussian.png')
            save_image(y.cpu(), img_output_dir + '/sample_recon_y.png')
            save_image(y_sample.cpu(), img_output_dir + '/sample_y.png')

            save_image(x_hat_iwt.cpu(), img_output_dir + '/sample_recon_x_iwt.png')
            save_image(x_sample_iwt.cpu(), img_output_dir + '/sample_z_iwt.png')
            save_image(x_sample_y_sample_iwt.cpu(), img_output_dir + '/sample_z_both_iwt.png')
            save_image(x_sample_y_sample_gaussian_iwt.cpu(), img_output_dir + '/sample_z_both_gaussian_iwt.png')
            save_image(x_y_sample_gaussian_iwt.cpu(), img_output_dir + '/sample_z_wt_gaussian_iwt.png')

            save_image(y_padded_iwt.cpu(), img_output_dir + '/sample_y_iwt.png')

            save_image(data128.cpu(), img_output_dir + '/sample128.png')
            save_image(data512.cpu(), img_output_dir + '/sample512.png')
            
    
    LOGGER.info('Full Model parameters: {}'.format(sum(x.numel() for x in full_model.wt_model.parameters()) + sum(x.numel() for x in full_model.iwt_model.parameters())))

    
    
    