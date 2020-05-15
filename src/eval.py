import time
import torch.nn.functional as F
from torchvision.utils import save_image
import wandb
from tqdm import tqdm, trange

from wt_utils import *

# Run model through dataloader and save all images
def eval_unet128(model, data_loader, args):
    model.eval()

    filters = create_filters(device=args.device)
    inv_filters = create_inv_filters(device=args.device)

    counter = 0

    for data, _ in tqdm(data_loader):
        data = data.to(args.device)
    
        Y = wt_128_3quads(data, filters, levels=3)

        # Get real 1st level masks
        Y_64 = Y[:, :, :64, :64]
        real_mask_64_tl, real_mask_64_tr, real_mask_64_bl, real_mask_64_br = get_4masks(Y_64, 32)
        Y_64_patches = torch.cat((real_mask_64_tl, real_mask_64_tr, real_mask_64_bl, real_mask_64_br), dim=1)

        with torch.no_grad():
            # Run through 128 mask network and get reconstructed image
            recon_mask_all = model(Y_64_patches)
            recon_mask_tr, recon_mask_bl, recon_mask_br = split_masks_from_channels(recon_mask_all)

        Y_real = wt(data, filters, levels=3)
        zeros = torch.zeros(recon_mask_tr.shape)
        
        real_img_128_padded = Y_real[:, :, :128, :128]
        real_img_128_padded = zero_pad(real_img_128_padded, 256, args.device)
        real_img_128_padded = iwt(real_img_128_padded, inv_filters, levels=3)

        # Collate all masks concatenated by channel to an image (slice up and put into a square)
        recon_mask_tr_img = collate_channels_to_img(recon_mask_tr, args.device)
        recon_mask_bl_img = collate_channels_to_img(recon_mask_bl, args.device)   
        recon_mask_br_img = collate_channels_to_img(recon_mask_br, args.device)
        
        recon_mask_tr_img = iwt(recon_mask_tr_img, inv_filters, levels=1)
        recon_mask_bl_img = iwt(recon_mask_bl_img, inv_filters, levels=1)    
        recon_mask_br_img = iwt(recon_mask_br_img, inv_filters, levels=1) 
        
        recon_mask_iwt = collate_patches_to_img(zeros, recon_mask_tr_img, recon_mask_bl_img, recon_mask_br_img)
        
        recon_mask_padded = zero_pad(recon_mask_iwt, 256, args.device)
        recon_mask_padded[:, :, :64, :64] = Y_64
        recon_img = iwt(recon_mask_padded, inv_filters, levels=3)
            
        # Save images
        for j in range(recon_img.shape[0]):
            save_image(recon_img.cpu(), args.output_dir + 'recon_img_{}.png'.format(counter))
            save_image(real_img_128_padded.cpu(), args.output_dir + 'img_128_{}.png'.format(counter))
            save_image(data.cpu(), args.output_dir + 'img_{}.png'.format(counter))

            counter += 1
