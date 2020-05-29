import torch
from torch.autograd import Variable
import numpy as np
import pywt

def load_UNET_checkpoint(model, optimizer, model_type, args):
    checkpoint = torch.load(args.output_dir + '/UNET_pixel_model_{}_itr{}.pth'.format(model_type, args.checkpoint), map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    logger = torch.load(args.output_dir + '/logger.pth')

    del checkpoint
    torch.cuda.empty_cache()

    return model, optimizer, logger

def load_UNET_weights(model, model_type, args):
    if model_type == '128':
        checkpoint_path = args.model_128_weights
    elif model_type == '256':
        checkpoint_path = args.model_256_weights
    else:
        raise ValueError("Improper model type")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    del checkpoint
    torch.cuda.empty_cache()

    return model


############### WT FUNCTIONS ###############
def wt(vimg, filters, levels=1):
    bs = vimg.shape[0]
    h = vimg.size(2)
    w = vimg.size(3)
    vimg = vimg.reshape(-1, 1, h, w)
    padded = torch.nn.functional.pad(vimg,(2,2,2,2))
    res = torch.nn.functional.conv2d(padded, Variable(filters[:,None]),stride=2)
    if levels>1:
        res[:,:1] = wt(res[:,:1], filters, levels-1)
        res[:,:1,32:,:] = res[:,:1,32:,:]*1.
        res[:,:1,:,32:] = res[:,:1,:,32:]*1.
        res[:,1:] = res[:,1:]*1.
    res = res.view(-1,2,h//2,w//2).transpose(1,2).contiguous().view(-1,1,h,w)
    return res.reshape(bs, -1, h, w)

def create_filters(device, wt_fn='bior2.2'):
    w = pywt.Wavelet(wt_fn)
    dec_hi = torch.Tensor(w.dec_hi[::-1]).to(device)
    dec_lo = torch.Tensor(w.dec_lo[::-1]).to(device)
    filters = torch.stack([dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1),
                           dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1),
                           dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1),
                           dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)], dim=0)
    return filters

def create_inv_filters(device, wt_fn='bior2.2'):
    w = pywt.Wavelet(wt_fn)

    rec_hi = torch.Tensor(w.rec_hi).to(device)
    rec_lo = torch.Tensor(w.rec_lo).to(device)
    
    inv_filters = torch.stack([rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)], dim=0)

    return inv_filters

def iwt(vres, inv_filters, levels=1):
    bs = vres.shape[0]
    h = vres.size(2)
    w = vres.size(3)
    vres = vres.reshape(-1, 1, h, w)
    res = vres.contiguous().view(-1, h//2, 2, w//2).transpose(1, 2).contiguous().view(-1, 4, h//2, w//2).clone()
    if levels > 1:
        res[:,:1] = iwt(res[:,:1], inv_filters, levels=levels-1)
    res = torch.nn.functional.conv_transpose2d(res, Variable(inv_filters[:,None]),stride=2)
    res = res[:,:,2:-2,2:-2] #removing padding

    return res.reshape(bs, -1, h, w)

############### COLLATE & PAD FUNCTIONS ###############

# Create padding on patch so that this patch is formed into a square image with other patches as 0
# 3 x 128 x 128 => 3 x target_dim x target_dim
def zero_pad(img, target_dim, device='cpu'):
    batch_size = img.shape[0]
    num_channels = img.shape[1]
    padded_img = torch.zeros((batch_size, num_channels, target_dim, target_dim), device=device)
    padded_img[:, :, :img.shape[2], :img.shape[3]] = img.to(device)
    
    return padded_img

def collate_patches_to_img(tl, tr, bl, br, device='cpu'):
    bs = tl.shape[0]
    c = tl.shape[1]
    h = tl.shape[2]
    w = tl.shape[3]
    
    frame = torch.empty((bs, c, 2*h, 2*w), device=device)
    frame[:, :, :h, :w] = tl.to(device)
    frame[:, :, :h, w:] = tr.to(device)
    frame[:, :, h:, :w] = bl.to(device)
    frame[:, :, h:, w:] = br.to(device)
    
    return frame

############### DATASET FUNCTIONS ###############
# data_variance = np.var(training_data.data / 255.0)
def calc_wt_var(training_loader, device='cpu'):
    mean_tl = 0.0
    meansq_tl = 0.0
    mean_tr = 0.0
    meansq_tr = 0.0
    mean_bl = 0.0
    meansq_bl = 0.0
    mean_br = 0.0
    meansq_br = 0.0

    count = 0

    filters = create_filters(device)

    for _, data in enumerate(training_loader):
        data_wt = wt(data.to(device), filters, levels=2)[:, :, :64, :64]
        tl = data_wt[:, :, :32, :32]
        tr = data_wt[:, :, :32, 32:]
        bl = data_wt[:, :, 32:, :32]
        br = data_wt[:, :, 32:, 32:]
        
        mean_tl += tl.sum()
        meansq_tl += (tl**2).sum()
        
        mean_tr += tr.sum()
        meansq_tr += (tr**2).sum()
        
        mean_bl += bl.sum()
        meansq_bl += (bl**2).sum()
        
        mean_br += br.sum()
        meansq_br += (br**2).sum()
        
        count += np.prod(data.shape)

    count /= 4
    total_mean_tl = mean_tl / count
    total_var_tl = (meansq_tl/count) - (total_mean_tl**2)

    total_mean_tr = mean_tr / count
    total_var_tr = (meansq_tr/count) - (total_mean_tr**2)

    total_mean_bl = mean_bl / count
    total_var_bl = (meansq_bl/count) - (total_mean_bl**2)

    total_mean_br = mean_br / count
    total_var_br = (meansq_br/count) - (total_mean_br**2)

    return total_var_tl, total_var_tr, total_var_bl, total_var_br