import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import pywt
import IPython
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
    h = mask.shape[2]
    w = mask.shape[3]

    inner_patch_h0 = h // (np.power(2, num_iwt-cur_iwt+1))
    inner_patch_w0 = w // (np.power(2, num_iwt-cur_iwt+1))
    outer_patch_h0 = inner_patch_h0 * 2
    outer_patch_w0 = inner_patch_w0 * 2

    mask[:, :, :inner_patch_h0, :inner_patch_w0].fill_(0)

    # Masking outer patches, only if we are not already at the edges of the image
    if outer_patch_h0 < h and outer_patch_w0 < w:
        mask[:, outer_patch_h0:, outer_patch_w0:].fill_(0)
    
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

# Normalize all values to range (0, 1)
# Minimum value in low frequency patches in training dataset = -1.9527
# Maximum value in low frequency patches in training dataset = 1.6612
def preprocess_low_freq(batch):
    low_freq = batch[:, :, 128:, 128:]
    low_freq = (low_freq + 1.953) / 3.795
    # low_freq = (low_freq + 1.953) * 100

    batch[:, :, 128:, 128:] = low_freq
    
    return batch

# Revert low frequency back to original range
def postprocess_low_freq(batch):
    low_freq = batch[:, :, 128:, 128:]
    low_freq = (low_freq * 3.795) - 1.953
    # low_freq = (low_freq / 100) - 1.953
    batch[:, :, 128:, 128:] = low_freq
    
    return batch

# Collates the high frequency patches to channels in the order of 1st, 3rd, and 4th quadrants
# Assumes number of wt = 1
def hf_collate_to_channels(wt_img, device='cpu'):
    h = wt_img.shape[2]
    w = wt_img.shape[3]
    first_quad = wt_img[:, :, :h // 2, w // 2:]
    third_quad = wt_img[:, :, h // 2:, :w // 2]
    fourth_quad = wt_img[:, :, h // 2:, w // 2:]

    return torch.cat((first_quad, third_quad, fourth_quad), dim=1).to(device)

# Collates the high frequency patches to channels in the order of 1st, 3rd, and 4th quadrants (smaller quadrants)
# Assumes number of wt = 2
def hf_collate_to_channels_wt2(wt_img, device='cpu'):
    h = wt_img.shape[2]
    w = wt_img.shape[3]
    inner_dim = h // 4
    outer_dim = inner_dim * 2
    first_quad = wt_img[:, :, :inner_dim, inner_dim:outer_dim]
    third_quad = wt_img[:, :, inner_dim:outer_dim, :inner_dim]
    fourth_quad = wt_img[:, :, inner_dim:outer_dim, inner_dim:outer_dim]
    
    return torch.cat((first_quad, third_quad, fourth_quad), dim=1).to(device)

# Collates high frequency patches back to image format with first patch = 0
# Assumes number of wt = 1
def hf_collate_to_img(wt_channels, device='cpu'):
    bs = wt_channels.shape[0]
    c = wt_channels.shape[1]
    h = wt_channels.shape[2] * 2
    w = wt_channels.shape[3] * 2
    
    wt_img = torch.zeros((bs, 3, h, w), device=device)
    wt_img[:, :, :h // 2, w // 2:] = wt_channels[:, :c // 3, :, :]
    wt_img[:, :, h // 2:, :w // 2] = wt_channels[:, c // 3: c // 3 * 2, :, :]
    wt_img[:, :, h // 2:, w // 2:] = wt_channels[:, c // 3 * 2:, :, :]

    return wt_img
