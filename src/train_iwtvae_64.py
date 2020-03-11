import os, sys
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
import numpy as np
from vae_models import IWTVAE_64, IWTVAE_64_Mask, WTVAE_64
from wt_datasets import CelebaDataset
from trainer import train_iwtvae
from arguments import args_parse
from utils.processing import zero_patches
import logging
import pywt
from random import sample


if __name__ == "__main__":
    # Accelerate training since fixed input sizes
    torch.backends.cudnn.benchmark = True 

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(message)s')
    LOGGER = logging.getLogger(__name__)

    args = args_parse()

    dataset_dir = os.path.join(args.root_dir, 'celeba64')
    train_dataset = CelebaDataset(dataset_dir, os.listdir(dataset_dir), WT=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=10, shuffle=True)
    sample_dataset = Subset(train_dataset, sample(range(len(train_dataset)), 8))
    sample_loader = DataLoader(sample_dataset, batch_size=8, shuffle=False) 
    
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        devices = ['cuda:0', 'cuda:1']
    else: 
        devices = ['cpu', 'cpu']

    if args.mask:
        iwt_model = IWTVAE_64_Mask(z_dim=args.z_dim, upsampling=args.upsampling, num_upsampling=args.num_upsampling, reuse=args.reuse)
        LOGGER.info('Running mask model')
    elif args.bottleneck_dim > 0:
        iwt_model = IWTVAE_64(z_dim=args.z_dim, bottleneck_dim=args.bottleneck_dim, upsampling='bottleneck', num_upsampling=args.num_upsampling, reuse=args.reuse)
        LOGGER.info('Running bottleneck model with dim = {}'.format(args.bottleneck_dim))
    else:
        iwt_model = IWTVAE_64(z_dim=args.z_dim, upsampling=args.upsampling, num_upsampling=args.num_upsampling, reuse=args.reuse)
        LOGGER.info('Running original model with upsampling = {}'.format(args.upsampling))
    
    if args.zero:
        LOGGER.info('Zero-ing out all patches other than 1st')

    iwt_model = iwt_model.to(devices[0])
    iwt_model.set_devices(devices)

    wt_model = WTVAE_64(z_dim=args.z_dim, num_wt=args.num_wt, unflatten=args.unflatten)
    wt_model.load_state_dict(torch.load(args.wtvae_model))
    wt_model.set_device(devices[1])
    wt_model.to(devices[1])
    wt_model.eval()
    
    train_losses = []
    optimizer = optim.Adam(iwt_model.parameters(), lr=args.lr)

    img_output_dir = os.path.join(args.root_dir, 'image_samples/iwtvae64_{}'.format(args.config))
    model_dir = os.path.join(args.root_dir, 'models/iwtvae64_{}/'.format(args.config))

    try:
        os.mkdir(img_output_dir)
        os.mkdir(model_dir)
    except:
        LOGGER.error('Could not make model & img output directories')
        raise Exception('Could not make model & img output directories')
    
    for epoch in range(1, args.epochs + 1):
        train_iwtvae(epoch, wt_model, iwt_model, optimizer, train_loader, train_losses, args)
        
        with torch.no_grad():
            iwt_model.eval()
            
            for data in sample_loader:
                data0 = data.to(devices[0])
                data1 = data.to(devices[1])
                
                z_sample = torch.randn(data.shape[0],100).to(devices[0])
                
                Y = wt_model(data1)[0]
                if args.zero:
                    Y = zero_patches(Y).to(devices[0])
                mu, var = iwt_model.encode(data0, Y)
                x_hat = iwt_model.decode(Y, mu)
                x_sample = iwt_model.decode(Y, z_sample)

                save_image(x_hat.cpu(), img_output_dir + '/sample_recon{}.png'.format(epoch))
                save_image(x_sample.cpu(), img_output_dir + '/sample_z{}.png'.format(epoch))
                save_image(Y.cpu(), img_output_dir + '/sample_y{}.png'.format(epoch))
                save_image(data.cpu(), img_output_dir + '/sample{}.png'.format(epoch))
    
        torch.save(iwt_model.state_dict(), model_dir + '/iwtvae_epoch{}.pth'.format(epoch))
    
    np.save(model_dir+'/train_losses.npy', train_losses)
    
    
    