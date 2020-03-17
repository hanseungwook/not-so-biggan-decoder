import os, sys
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import numpy as np
from vae_models import WTCNN_512, wt
from wt_datasets import CelebaDataset
from trainer import train_wtcnn
from arguments import args_parse
from .utils.utils import set_seed, save_plot, zero_pad, create_filters, create_inv_filters
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
    wt_model = WTCNN_512()
    wt_model = wt_model.to(device)
    wt_model.set_filters(filters)
    wt_model.set_device(device)
    
    train_losses = []
    optimizer = optim.Adam(wt_model.parameters(), lr=args.lr)

    # Create output directories
    img_output_dir = os.path.join(args.root_dir, 'wtvae_results/image_samples/wtcnn512_{}'.format(args.config))
    model_dir = os.path.join(args.root_dir, 'wtvae_results/models/wtcnn512_{}/'.format(args.config))

    try:
        os.mkdir(img_output_dir)
        os.mkdir(model_dir)
    except:
        LOGGER.error('Could not make model & img output directories')
        raise Exception('Could not make model & img output directories')
    
    for epoch in range(1, args.epochs + 1):
        train_wtcnn(epoch, wt_model, optimizer, train_loader, train_losses, args, writer)
        
        with torch.no_grad():
            wt_model.eval()
            
            for data in sample_loader:
                x = data.clone().detach()
                wt_data = wt_model(data.to(device))
                
                x_wt = wt(x.reshape(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3]), wt_model.filters, levels=2)
                x_wt = x_wt.reshape(x.shape)
                x_wt = x_wt[:, :, :128, :128]
                
                save_image(wt_data.cpu(), img_output_dir + '/sample_recon_y{}.png'.format(epoch))
                save_image(x_wt.cpu(), img_output_dir + '/sample{}.png'.format(epoch))
    
        torch.save(wt_model.state_dict(), model_dir + '/wtvae_epoch{}.pth'.format(epoch))
    
    # Save train losses and plot
    np.save(model_dir+'/train_losses.npy', train_losses)
    save_plot(train_losses, img_output_dir + '/train_loss.png')
    
    writer.close()
    LOGGER.info('WT Model parameters: {}'.format(sum(x.numel() for x in wt_model.parameters())))

    
    
    