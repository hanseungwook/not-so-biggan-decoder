import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
from vae_models import WTVAE_64, iwt
from wt_datasets import CelebaDataset
from trainer import train_wtvae
from arguments import args_parse
import logging
import pywt


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s')
    LOGGER = logging.getLogger(__name__)

    args = args_parse()

    dataset_dir = os.path.join(args.root_dir, 'celeba64')
    train_dataset = CelebaDataset(dataset_dir, os.listdir(dataset_dir), WT=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=10, shuffle=True)
    
    DEVICE = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info('Device: {}'.format(DEVICE))

    model = WTVAE_64(z_dim=args.z_dim, num_wt=args.num_wt)
    model = model.to(DEVICE)

    w = pywt.Wavelet('bior2.2')
    rec_hi = torch.Tensor(w.rec_hi).to(DEVICE)
    rec_lo = torch.Tensor(w.rec_lo).to(DEVICE)

    inv_filters = torch.stack([rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1),
                                rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1),
                                rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1),
                                rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)], dim=0)

    model.set_inv_filters(inv_filters)
    model.set_device(DEVICE)
    
    train_losses = []
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    img_output_dir = os.path.join(args.root_dir, 'celeba_wtvae64_')
    model_dir = os.path.join(args.root_dir, 'wtvae64_' + args.config + '_models/')

    try:
        os.mkdir(img_output_dir)
        os.mkdir(model_dir)
    except:
        LOGGER.error('Cannot make model & img output directories')
    
    for epoch in range(1, args.epochs + 1):
        train_wtvae(epoch, model, optimizer, train_loader, train_losses, args)

        # 32 sets of random ZDIMS-float vectors
        sample = torch.randn(32, 100)
        sample = sample.to(DEVICE)

        x_sample1 = model.decode(sample)
        save_image(x_sample1[:8].cpu(), img_output_dir + '/decoded_sample{}.png'.format(epoch))
        
        x_sample1 = x_sample1.view(-1,1,64,64)
        x_sample1 = iwt(x_sample1, inv_filters, levels=3)
        x_sample1 = x_sample1.view(-1,3,64,64)
        x_sample1 = x_sample1.contiguous()

        save_image(x_sample1[:8].cpu(), img_output_dir + '/sample{}.png'.format(epoch))
        torch.save(model.state_dict(), model_dir + '/wtvae_epoch{}.pth'.format(epoch))
    
    np.save(model_dir+'/train_losses.npy', train_losses)
    
    
    