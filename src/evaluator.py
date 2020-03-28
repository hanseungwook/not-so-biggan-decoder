import torch
from torchvision.utils import save_image
import numpy as np
from vae_models import wt
from utils.utils import zero_mask, zero_pad, save_plot, hf_collate_to_img, hf_collate_to_channels, hf_collate_to_channels_wt2

def eval_wtvae_pair(epoch, model, sample_loader, args, img_output_dir, model_dir):
    with torch.no_grad():
        model.eval()
        
        for data in sample_loader:
            if model.cuda:
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
