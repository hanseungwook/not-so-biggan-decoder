import os, sys
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import numpy as np
from vae_models import WTVAE_128_1, IWTVAE_512_Mask_2, FullVAE_512, wt, WT, iwt, IWT
from wt_datasets import CelebaDatasetPair
from trainer import train_fullvae
from arguments import args_parse
from utils.utils import set_seed, save_plot, zero_pad, create_filters, create_inv_filters, zero_mask
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
    sample_dataset = Subset(train_dataset, sample(range(len(train_dataset)), 32))
    sample_loader = DataLoader(sample_dataset, batch_size=32, shuffle=False) 
    
    if args.device >= 0:
        device = 'cuda:{}'.format(args.device)
    else: 
        device = 'cpu'
    print('Device: {}'.format(device))

    # Setting up WT & IWT filters
    filters = create_filters(device=device)
    inv_filters = create_inv_filters(device=device)

    wt_model = WTVAE_128_1(z_dim=args.z_dim_wt, num_wt=args.num_iwt)
    wt_model.set_filters(filters)
    wt_model = wt_model.to(device)
    iwt_model = IWTVAE_512_Mask_2(z_dim=args.z_dim_iwt, num_iwt=args.num_iwt)
    iwt_model.set_filters(inv_filters)
    iwt_model = iwt_model.to(device)

    wt_fn = WT(wt=wt, num_wt=args.num_iwt)
    wt_fn.set_filters(filters)
    iwt_fn = IWT(iwt=iwt, num_iwt=args.num_iwt)
    iwt_fn.set_filters(inv_filters)

    # Given saved model, load and freeze model
    if args.iwt_model and args.wt_model:
        iwt_checkpoint = torch.load(args.iwt_model, map_location=device)
        iwt_model.load_state_dict(iwt_checkpoint['model_state_dict'])
        wt_model.load_state_dict(torch.load(args.wt_model, map_location=device))

    img_output_dir = os.path.join(args.root_dir, 'wtvae_results/image_samples/fullvae128512_eval{}'.format(args.config))
    model_dir = os.path.join(args.root_dir, 'wtvae_results/models/fullvae128512_eval{}/'.format(args.config))

    try:
        os.mkdir(img_output_dir)
        os.mkdir(model_dir)
    except:
        LOGGER.error('Could not make model & img output directories')
        raise Exception('Could not make model & img output directories')
            
    with torch.no_grad():
        wt_model.eval()
        iwt_model.eval()
        
        for data in sample_loader:
            data128 = data[0].to(device)
            data512 = data[1].to(device)
            z, mu_wt, logvar_wt = wt_model.encode(data128)

            # Creating z sample for WT model by adding Gaussian noise ~ N(0,1)
            z_sample1 = torch.randn(z.shape).to(device)
            z_sample2 = z + torch.randn(z.shape).to(device)

            y = wt_model.decode(z)
            y_sample1 = wt_model.decode(z_sample1)
            y_sample2 = wt_model.decode(z_sample2)

            y_padded = zero_pad(y, target_dim=512, device=device)
            y_sample_padded1 = zero_pad(y_sample1, target_dim=512, device=device)
            y_sample_padded2 = zero_pad(y_sample2, target_dim=512, device=device)
            
            data512_wt = wt_fn(data512)
            # Zero out first patch and apply IWT
            data512_mask = zero_mask(data512_wt, args.num_iwt, 1)
            data512_mask = iwt_fn(data512_wt)

            mask, mu, var = iwt_model(data512_mask)

            mask_wt = wt_fn(mask)

            img_low = iwt_fn(y_padded)
            img_low_sample1 = iwt_fn(y_sample_padded1)
            img_low_sample2 = iwt_fn(y_sample_padded2)

            img_recon = iwt_fn(y_padded + mask_wt)
            img_sample1_recon = iwt_fn(y_sample_padded1 + mask_wt)
            img_sample2_recon = iwt_fn(y_sample_padded2 + mask_wt)


            # Save images
            save_image(y.cpu(), img_output_dir + '/recon_y.png')
            save_image(y_sample1.cpu(), img_output_dir + '/sample1_y.png')
            save_image(y_sample2.cpu(), img_output_dir + '/sample2_y.png')
            save_image(data512_mask.cpu(), img_output_dir + '/mask.png')
            save_image(img_low.cpu(), img_output_dir + '/low_img.png')
            save_image(img_low_sample1.cpu(), img_output_dir + '/low_img_sample1.png')
            save_image(img_low_sample2.cpu(), img_output_dir + '/low_img_sample2.png')
            save_image(img_recon.cpu(), img_output_dir + '/recon_img.png')
            save_image(img_sample1_recon.cpu(), img_output_dir + '/sample1_img.png')
            save_image(img_sample2_recon.cpu(), img_output_dir + '/sample2_img.png')
            save_image(data512.cpu(), img_output_dir + '/img.png')
            
    
    LOGGER.info('Full Model parameters: {}'.format(sum(x.numel() for x in wt_model.parameters()) + sum(x.numel() for x in iwt_model.parameters())))

    
    
    