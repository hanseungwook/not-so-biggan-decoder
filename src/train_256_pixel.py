import torch
import time
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
import wandb
from UNET_utils import *
from tqdm import tqdm, trange

import sys
import os
sys.path.insert(0, 'Pytorch-UNet')
sys.path.insert(0, 'utils')
import numpy as np
import matplotlib.pyplot as plt
from plotting_utils import plot_pytorch_images, make_grid
import torchvision.datasets as dset
import torchvision.utils as vutils
import torchvision
from misc import merge
from unet.unet_model import UNet
from torch.nn import functional as F

loss_func = torch.nn.MSELoss(reduction='sum')
def train_256(epoch, state_dict, model, optimizer, train_loader, valid_loader, args, logger):
    model.train()
    
    # Train loop
    for data in tqdm(train_loader):
        start_time = time.time() 
        optimizer.zero_grad()

        #Downsample then reconstruct upsampled version
        x = data[0]
        y = F.interpolate(F.interpolate(x, args.low_resolution, mode="bilinear"), args.image_size, mode="bilinear")
        x = x.to(args.device)
        y = y.to(args.device)
        x_mask = x - y 
        x_mask_hat = model(y)
        x_hat = y + x_mask_hat

        #Compute loss and take step
        loss = loss_func(x_mask_hat, x_mask)
        loss.backward()
        optimizer.step()

        # Calculate iteration time
        end_time = time.time()
        itr_time = end_time - start_time

        # Update logger & wandb
        logger.update(state_dict['itr'], loss.cpu().item(), itr_time)
        wandb.log({'train_loss': loss.item()}, commit=False)
        wandb.log({'train_itr_time': itr_time}, commit=True)    

        # Save images, logger, weights on save_every interval
        if not state_dict['itr'] % args.save_every:
            # Save images
            save_image(x_mask.cpu(), args.output_dir + 'train_real_mask_itr{}.png'.format(state_dict['itr']))
            save_image(x_mask_hat.cpu(), args.output_dir + 'train_recon_mask_itr{}.png'.format(state_dict['itr']))
            save_image(y.cpu(), args.output_dir + 'train_low_img_256_itr{}.png'.format(state_dict['itr']))
            save_image(x_hat.cpu(), args.output_dir + 'train_recon_img_256_itr{}.png'.format(state_dict['itr']))
            save_image(x.cpu(), args.output_dir + 'train_real_img_256_itr{}.png'.format(state_dict['itr']))

            # Save model & optimizer weights
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, args.output_dir + '/UNET_pixel_model_256_itr{}.pth'.format(state_dict['itr']))
            
            # Save logger 
            torch.save(logger, args.output_dir + '/logger.pth')
        
        if not state_dict['itr'] % args.valid_every and state_dict['itr'] != 0:
                print("here")
                model.eval()
                val_losses = []

                with torch.no_grad(): 
                    for data in tqdm(valid_loader):
                        print("here")
                        x_val = data[0]
                        y_val = F.interpolate(F.interpolate(x_val, args.low_resolution, mode="bilinear"), args.image_size, mode="bilinear")
                        x_val = x_val.to(args.device)
                        y_val = y_val.to(args.device)    
                        x_mask_val = x_val - y_val
                        x_mask_hat_val = model(y_val)
                        x_hat_val = y_val + x_mask_hat_val
                        loss_val = loss_func(x_mask_hat_val, x_mask_val)
                        val_losses.append(loss_val.item())

                    save_image(x_mask_val.cpu(), args.output_dir + 'val_real_mask_itr{}.png'.format(state_dict['itr']))
                    save_image(x_mask_hat_val.cpu(), args.output_dir + 'val_recon_mask_itr{}.png'.format(state_dict['itr']))
                    save_image(y_val.cpu(), args.output_dir + 'val_low_img_256_itr{}.png'.format(state_dict['itr']))
                    save_image(x_hat_val.cpu(), args.output_dir + 'val_recon_img_256_itr{}.png'.format(state_dict['itr']))
                    save_image(x_val.cpu(), args.output_dir + 'val_real_img_256_itr{}.png'.format(state_dict['itr']))

                    val_losses_mean = np.mean(val_losses)
                    wandb.log({'val_loss': val_losses_mean}, commit=True)
                    logger.update_val_loss(state_dict['itr'], val_losses_mean)
                    val_losses.clear()
            
                model.train()
        # Increment iteration number
        state_dict['itr'] += 1       