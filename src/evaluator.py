import torch
from torchvision.utils import save_image
import numpy as np
from vae_models import wt
from utils.utils import zero_mask, zero_patches, zero_pad, save_plot, hf_collate_to_img, hf_collate_to_channels, hf_collate_to_channels_wt2, collate_masks_to_img

def eval_wtvae_pair(epoch, model, sample_loader, args, img_output_dir, model_dir):
    with torch.no_grad():
        model.eval()
        
        for data in sample_loader:
            data0 = data[0].to(model.device)
            data1 = data[1].to(model.device)
                
            # Run encoder: get z and sampled z
            z_sample1 = torch.randn(data1.shape[0], args.z_dim).to(model.device)
            z, mu_wt, logvar_wt = model.encode(data0)

            # Run decoder: get y and sampled y
            y = model.decode(z)
            y_sample = model.decode(z_sample1)

            # Create padded versions
            target_dim = np.power(2, args.num_wt) * y.shape[2]
            y_padded = zero_pad(y, target_dim=target_dim, device=model.device)
            y_sample_padded = zero_pad(y_sample, target_dim=target_dim, device=model.device)

            x_wt = wt(data1, model.filters, levels=args.num_wt)
            x_wt = x_wt[:, :, :y.shape[2], :y.shape[3]]
            
            save_image(y_padded.cpu(), img_output_dir + '/recon_y_padded{}.png'.format(epoch))
            save_image(y.cpu(), img_output_dir + '/recon_y{}.png'.format(epoch))
            save_image(y_sample.cpu(), img_output_dir + '/sample_y{}.png'.format(epoch))
            save_image(x_wt.cpu(), img_output_dir + '/target{}.png'.format(epoch))

    torch.save(model.state_dict(), model_dir + '/wtvae_epoch{}.pth'.format(epoch))

def eval_iwtvae(epoch, wt_model, iwt_model, optimizer, iwt_fn, sample_loader, args, img_output_dir, model_dir, writer):
    with torch.no_grad():
        iwt_model.eval()
        
        for data in sample_loader:
            data = data.to(wt_model.device)
            
            # Applying WT to X to get Y
            Y = wt_model(data)
            save_image(Y.cpu(), img_output_dir + '/sample_y_before_zero{}.png'.format(epoch))
            Y_full = Y.clone()

            # Zero-ing out rest of the patches
            if args.zero:
                Y = zero_patches(Y, num_wt=args.num_iwt)

            # Get sample
            z_sample = torch.randn(data.shape[0], args.z_dim).to(iwt_model.device)

            # Encoder
            mu, var = iwt_model.encode(Y_full - Y)

            # Decoder -- two versions, real z and asmple z
            mask = iwt_model.decode(Y, mu)
            mask = zero_mask(mask, args.num_iwt, 1)
            assert (mask[:, :, :128, :128] == 0).all()

            mask_sample = iwt_model.decode(Y, z_sample)
            mask_sample = zero_mask(mask_sample, args.num_iwt, 1)
            assert (mask_sample[:, :, :128, :128] == 0).all()

            # Construct x_wt_hat and x_wt_hat_sample and apply IWT to get reconstructed and sampled images
            x_wt_hat = Y + mask
            x_wt_hat_sample = Y + mask_sample

            x_hat = iwt_fn(x_wt_hat)
            x_sample = iwt_fn(x_wt_hat_sample)
            
            # Save images
            save_image(x_hat.cpu(), img_output_dir + '/recon_x{}.png'.format(epoch))
            save_image(x_sample.cpu(), img_output_dir + '/sample_x{}.png'.format(epoch))
            save_image(x_wt_hat.cpu(), img_output_dir + '/recon_x_wt{}.png'.format(epoch))
            save_image(x_wt_hat_sample.cpu(), img_output_dir + '/sample_x_wt{}.png'.format(epoch))
            save_image((Y_full-Y).cpu(), img_output_dir + '/encoder_input{}.png'.format(epoch))
            save_image(Y.cpu(), img_output_dir + '/y{}.png'.format(epoch))
            save_image(data.cpu(), img_output_dir + '/target{}.png'.format(epoch))

    torch.save({
                'epoch': epoch,
                'model_state_dict': iwt_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, model_dir + '/iwtvae_epoch{}.pth'.format(epoch))

def eval_iwtvae_3masks(epoch, wt_model, iwt_model, optimizer, iwt_fn, sample_loader, args, img_output_dir, model_dir, writer):
    with torch.no_grad():
        iwt_model.eval()
        
        for data in sample_loader:
            data = data.to(wt_model.device)
            
            # Applying WT to X to get Y
            Y = wt_model(data)
            low = Y[:, :, :128, :128]
            mask1 = Y[:, :, :128, 128:256]
            mask2 = Y[:, :, 128:256, :128]
            mask3 = Y[:, :, 128:256, 128:256]
            masks = torch.cat((mask1, mask2, mask3), dim=1)

            # Encoder
            mu, var = iwt_model.encode(masks)

            # Decoder -- two versions, real z and asmple z
            mask1_hat, mask2_hat, mask3_hat = iwt_model.decode(mu)
            mask1_sample_hat, mask2_sample_hat, mask3_sample_hat = iwt_model.sample(data.shape[0])
            
            # Collate 3 masks into 1 image
            masks_target = Y
            masks_hat = collate_masks_to_img(low, mask1_hat, mask2_hat, mask3_hat)
            masks_sample_hat = collate_masks_to_img(low, mask1_sample_hat, mask2_sample_hat, mask3_sample_hat)

            img_hat = iwt_fn(masks_hat)
            img_sample_hat = iwt_fn(masks_sample_hat)

            # Save images
            save_image(masks_target.cpu(), img_output_dir + '/y{}.png'.format(epoch))
            save_image(masks_hat.cpu(), img_output_dir + '/recon_y{}.png'.format(epoch))
            save_image(masks_sample_hat.cpu(), img_output_dir + '/sample_y{}.png'.format(epoch))
            save_image(img_hat.cpu(), img_output_dir + '/recon_img{}.png'.format(epoch))
            save_image(img_sample_hat.cpu(), img_output_dir + '/sample_img{}.png'.format(epoch))
            save_image(data.cpu(), img_output_dir + '/img{}.png'.format(epoch))

    torch.save({
                'epoch': epoch,
                'model_state_dict': iwt_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
               }, model_dir + '/iwtvae_epoch{}.pth'.format(epoch))


def eval_iwtvae_iwtmask(epoch, wt_model, iwt_model, optimizer, iwt_fn, sample_loader, args, img_output_dir, model_dir, writer):
    with torch.no_grad():
        iwt_model.eval()
        
        for data in sample_loader:
            data = data.to(wt_model.device)
            
            # Applying WT to X to get Y
            Y = wt_model(data)
            Y_full = Y.clone()
        
            # Zeroing out first patch
            Y = zero_mask(Y, args.num_iwt, 1)

            # IWT all the leftover high frequencies
            Y = iwt_fn(Y)

            # Get sample
            z_sample = torch.randn(data.shape[0], args.z_dim).to(iwt_model.device)

            # Encoder
            mu, var = iwt_model.encode(Y)

            # Decoder -- two versions, real z and asmple z
            mask = iwt_model.decode(mu)
            mask_sample = iwt_model.decode(z_sample)

            mask_wt = wt_model(mask)
            mask_sample_wt = wt_model(mask_sample)

            mask_wt[:, :, :128, :128] += Y_full[:, :, :128, :128]
            mask_sample_wt[:, :, :128, :128] += Y_full[:, :, :128, :128]
            padded = torch.zeros(Y.shape, device=Y_full.device)
            padded[:, :, :128, :128] = Y_full[:, :, :128, :128]

            img_low = iwt_fn(padded)
            img_recon = iwt_fn(mask_wt)
            img_sample_recon = iwt_fn(mask_sample_wt)
            
            # Save images
            save_image(Y.cpu(), img_output_dir + '/y{}.png'.format(epoch))
            save_image(mask.cpu(), img_output_dir + '/recon_y{}.png'.format(epoch))
            save_image(mask_sample.cpu(), img_output_dir + '/sample_y{}.png'.format(epoch))
            save_image(img_low.cpu(), img_output_dir + '/low_img{}.png'.format(epoch))
            save_image(img_recon.cpu(), img_output_dir + '/recon_img{}.png'.format(epoch))
            save_image(img_sample_recon.cpu(), img_output_dir + '/recon_sample_img{}.png'.format(epoch))
            save_image(data.cpu(), img_output_dir + '/target{}.png'.format(epoch))

    torch.save({
                'epoch': epoch,
                'model_state_dict': iwt_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
               }, model_dir + '/iwtvae_epoch{}.pth'.format(epoch))

# Evaluation method for AE that takes in IWT of first patch to produce mask
def eval_iwtae_iwtmask(epoch, wt_model, iwt_model, optimizer, iwt_fn, sample_loader, args, img_output_dir, model_dir, writer, save=True):
    with torch.no_grad():
        iwt_model.eval()
        
        for data in sample_loader:
            data = data.to(wt_model.device)
            
            # Applying WT to X to get Y
            Y = wt_model(data)
            
            # Zeroing out first patch, if given zero arg
            Y_mask = zero_mask(Y, args.num_iwt, 1)
            # IWT all the leftover high frequencies
            Y_mask = iwt_fn(Y_mask)

            # Getting IWT of only first patch
            Y_low = zero_patches(Y, args.num_iwt)
            Y_low = iwt_fn(Y_low)

            # Run model to get mask (zero out first patch of mask) and x_wt_hat
            mask, mu, var = iwt_model(Y_low)

            # Add first patch to WT'ed mask
            mask_wt = wt_model(mask)
            mask_wt[:, :, :128, :128] += Y[:, :, :128, :128]

            img_recon = iwt_fn(mask_wt)
            
            # Save images
            save_image(Y_low.cpu(), img_output_dir + '/y{}.png'.format(epoch))
            save_image(mask.cpu(), img_output_dir + '/recon_mask{}.png'.format(epoch))
            save_image(Y_mask.cpu(), img_output_dir + '/mask{}.png'.format(epoch))
            save_image(img_recon.cpu(), img_output_dir + '/recon_img{}.png'.format(epoch))
            save_image(data.cpu(), img_output_dir + '/img{}.png'.format(epoch))            

    if save:
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': iwt_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, model_dir + '/iwtvae_epoch{}.pth'.format(epoch))

def eval_full_wtvae128_iwtae512(epoch, full_model, optimizer, sample_loader, args, img_output_dir, model_dir, writer, save=True):
    # toggle model to train mode, IWT model in eval b/c frozen
    full_model.eval()
    full_model.wt_model.eval()
    full_model.iwt_model.eval()
    
    for batch_idx, data in enumerate(sample_loader):
        optimizer.zero_grad()
        
        X_128, X_512 = data

        Y_low_hat, mask_hat, X_hat, mu, logvar = full_model(X_128)
        Y_low_sample_hat, mask_sample_hat, X_sample_hat = full_model.sample(X_128.shape[0])
        Y_low = full_model.wt_model.wt(X_128.to(full_model.wt_model.device))[:, :, :128, :128]
        X_low = full_model.iwt_model.iwt(Y_low.to(full_model.iwt_model.device))
        X_wt = full_model.wt_model.wt(X_512.to(full_model.wt_model.device))

        # Save images
        save_image(Y_low_hat.cpu(), img_output_dir + '/y_wt_recon{}.png'.format(epoch))
        save_image(mask_hat.cpu(), img_output_dir + '/mask_recon{}.png'.format(epoch))
        save_image(X_hat.cpu(), img_output_dir + '/X_recon{}.png'.format(epoch))

        save_image(Y_low_sample_hat.cpu(), img_output_dir + '/y_wt_sample{}.png'.format(epoch))
        save_image(mask_sample_hat.cpu(), img_output_dir + '/mask_sample{}.png'.format(epoch))
        save_image(X_sample_hat.cpu(), img_output_dir + '/X_sample{}.png'.format(epoch))

        save_image(Y_low.cpu(), img_output_dir + 'y_wt{}.png'.format(epoch))
        save_image(X_512.cpu(), img_output_dir + '/X{}.png'.format(epoch))
        save_image(X_low.cpu(), img_output_dir + '/X_low{}.png'.format(epoch))
        save_image(X_wt.cpu(), img_output_dir + '/X_wt{}.png'.format(epoch))

    if save:
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': iwt_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, model_dir + '/fullvae_epoch{}.pth'.format(epoch))
        

def eval_ae_mask(epoch, wt_model, model, sample_loader, args, img_output_dir, model_dir):
    with torch.no_grad():
        model.eval()
        
        for data in sample_loader:
            data = data.to(model.device)
            
            # Get Y
            Y = wt_model(data)
            
            # Zeroing out first patch
            Y = zero_mask(Y, num_iwt=args.num_wt, cur_iwt=1)

            x_hat = model(Y.to(model.device))

            save_image(x_hat.cpu(), img_output_dir + '/sample_recon{}.png'.format(epoch))
            save_image(Y.cpu(), img_output_dir + '/sample{}.png'.format(epoch))

    torch.save(model.state_dict(), model_dir + '/aemask_epoch{}.pth'.format(epoch))

def eval_ae_mask_channels(epoch, wt_model, model, sample_loader, args, img_output_dir, model_dir):
    with torch.no_grad():
        model.eval()
        
        for data in sample_loader:
            data = data.to(model.device)
            
            # Get Y
            Y = wt_model(data)
            
            # Zeroing out first patch
            Y = zero_mask(Y, num_iwt=args.num_wt, cur_iwt=1)
            if args.num_wt == 1:
                Y = hf_collate_to_channels(Y, device=model.device)
            elif args.num_wt == 2:
                Y = hf_collate_to_channels_wt2(Y, device=model.device)
            
            x_hat = model(Y.to(model.device))
            x_hat = hf_collate_to_img(x_hat)
            Y = hf_collate_to_img(Y)

            save_image(x_hat.cpu(), img_output_dir + '/sample_recon{}.png'.format(epoch))
            save_image(Y.cpu(), img_output_dir + '/sample{}.png'.format(epoch))

    torch.save(model.state_dict(), model_dir + '/aemask_epoch{}.pth'.format(epoch))
