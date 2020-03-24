import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import pywt

# Zeroing out all other patches than the first for WT image: 4D: B * C * H * W
def zero_patches(img, num_wt):
    padded = torch.zeros(img.shape)
    patch_dim = img.shape[2] // np.power(2, num_wt)
    padded[:, :, :patch_dim, :patch_dim] = img[:, :, :patch_dim, :patch_dim]
    
    return padded

# Create padding on patch so that this patch is formed into a square image with other patches as 0
# 3 x 128 x 128 => 3 x target_dim x target_dim
def zero_pad(img, target_dim, device='cpu'):
    batch_size = img.shape[0]
    num_channels = img.shape[1]
    padded_img = torch.zeros((batch_size, num_channels, target_dim, target_dim)).to(device)
    padded_img[:, :, :img.shape[2], :img.shape[3]] = img.to(device)
    
    return padded_img
    

# Zeroing out the first patch's portion of the mask
def zero_mask(mask, num_iwt, cur_iwt):
    h = mask.shape[1]
    w = mask.shape[2]

    inner_patch_h0 = h // (np.power(2, num_iwt-cur_iwt+1))
    inner_patch_w0 = w // (np.power(2, num_iwt-cur_iwt+1))
    outer_patch_h0 = inner_patch_h0 * 2
    outer_patch_w0 = inner_patch_w0 * 2

    mask[:, :inner_patch_h0, :inner_patch_w0].fill_(0)

    # Masking outer patches, only if we are not already at the edges of the image
    if outer_patch_h0 < h and outer_patch_w0 < w:
        mask[:, outer_patch_h0:, outer_patch_w0:].fill_(0)

    # mask[:, :h//(np.power(2, num_iwt)), :w//np.power(2, num_iwt)].fill_(0)
    
    return mask

def set_seed(seed, cudnn=True):
    """
    Seed everything we can!
    Note that gym environments might need additional seeding (env.seed(seed)),
    and num_workers needs to be set to 1.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # note: the below slows down the code but makes it reproducible
    if (seed is not None) and cudnn:
        torch.backends.cudnn.deterministic = True

def save_plot(data, save_file):
    plt.plot(data)
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.title('Train Loss')
    plt.savefig(save_file)

def create_filters(device):
    w = pywt.Wavelet('bior2.2')

    dec_hi = torch.Tensor(w.dec_hi[::-1]).to(device)
    dec_lo = torch.Tensor(w.dec_lo[::-1]).to(device)

    filters = torch.stack([dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1),
                           dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1),
                           dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1),
                           dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)], dim=0)

    return filters

def create_inv_filters(device):
    w = pywt.Wavelet('bior2.2')

    rec_hi = torch.Tensor(w.rec_hi).to(device)
    rec_lo = torch.Tensor(w.rec_lo).to(device)
    
    inv_filters = torch.stack([rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)], dim=0)

    return inv_filters

def calc_grad_norm_2(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    return total_norm