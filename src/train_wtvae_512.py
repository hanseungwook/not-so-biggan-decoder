import os, sys
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import numpy as np
from vae_models import WTVAE_512, WTVAE_512_1, wt
from wt_datasets import CelebaDataset
from trainer import train_wtvae
from arguments import args_parse
from utils.utils import set_seed, save_plot, zero_pad, create_filters, create_inv_filters
import matplotlib.pyplot as plt
import logging
import pywt
from random import sample


if __name__ == "__main__":
    # Accelerate training since fixed input sizes
    torch.backends.cudnn.benchmark = True 

    # Setting up logger
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(message)s')
    LOGGER = logging.getLogger(__name__)

    args = args_parse()

    # Setting up tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.root_dir, 'runs'))

    # Set seed
    set_seed(args.seed)

    # Create training and sample dataset (to test out model and save images for)
    dataset_dir = os.path.join(args.root_dir, 'data/celebaHQ512')
    dataset_files = sample(os.listdir(dataset_dir), 10000)
    train_dataset = CelebaDataset(dataset_dir, dataset_files, WT=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=10, shuffle=True)
    sample_dataset = Subset(train_dataset, sample(range(len(train_dataset)), 8))
    sample_loader = DataLoader(sample_dataset, batch_size=8, shuffle=False) 
    
    if torch.cuda.is_available():
        device = 'cuda:0'
    else: 
        device = 'cpu'

    # Setting up WT & IWT filters
    filters = create_filters(device=device)

    # Create model, set filters for WT (calculating loss), and set device
    wt_model = WTVAE_512_1(z_dim=args.z_dim, num_wt=args.num_iwt)
    wt_model = wt_model.to(device)
    wt_model.set_filters(filters)
    wt_model.set_device(device)
    
    train_losses = []
    optimizer = optim.Adam(wt_model.parameters(), lr=args.lr)

    # Create output directories
    img_output_dir = os.path.join(args.root_dir, 'wtvae_results/image_samples/wtvae512_{}'.format(args.config))
    model_dir = os.path.join(args.root_dir, 'wtvae_results/models/wtvae512_{}/'.format(args.config))

    try:
        os.mkdir(img_output_dir)
        os.mkdir(model_dir)
    except:
        LOGGER.error('Could not make model & img output directories')
        raise Exception('Could not make model & img output directories')

    for epoch in range(1, args.epochs + 1):
        # Setting current kl weight to start weight
        args.kl_weight = args.kl_start

        train_wtvae(epoch, wt_model, optimizer, train_loader, train_losses, args, writer)
        
        with torch.no_grad():
            wt_model.eval()
            
            for data in sample_loader:
                z_sample1 = torch.randn(data.shape[0], args.z_dim).to(device)
                x = data.clone().detach().to(device)

                # z, mu_wt, logvar_wt, m1_idx, m2_idx = wt_model.encode(data.to(device))
                # y = wt_model.decode(z, m1_idx, m2_idx)
                # y_sample = wt_model.decode(z_sample1, m1_idx, m2_idx)
                
                z, mu_wt, logvar_wt = wt_model.encode(data.to(device))
                y = wt_model.decode(z)
                y_sample = wt_model.decode(z_sample1)

                y_padded = zero_pad(y, target_dim=512, device=device)
                y_sample_padded = zero_pad(y_sample, target_dim=512, device=device)

                x_wt = wt(x.reshape(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3]), wt_model.filters, levels=2)
                x_wt = x_wt.reshape(x.shape)
                x_wt = x_wt[:, :, :128, :128]
                
                save_image(y_padded.cpu(), img_output_dir + '/sample_padded_y{}.png'.format(epoch))
                save_image(y.cpu(), img_output_dir + '/sample_recon_y{}.png'.format(epoch))
                save_image(y_sample.cpu(), img_output_dir + '/sample_y{}.png'.format(epoch))
                save_image(x_wt.cpu(), img_output_dir + '/sample{}.png'.format(epoch))
    
        torch.save(wt_model.state_dict(), model_dir + '/wtvae_epoch{}.pth'.format(epoch))
    
    # Save train losses and plot
    np.save(model_dir+'/train_losses.npy', train_losses)
    save_plot(train_losses, img_output_dir + '/train_loss.png')
    
    writer.close()
    LOGGER.info('WT Model parameters: {}'.format(sum(x.numel() for x in wt_model.parameters())))

    
    
    