import torch
import numpy as np
import matplotlib.pyplot as plt
import random

# Zeroing out all other patches than the first for WT image: 4D: B * C * H * W
def zero_patches(img, num_wt):
    patch_dim = img.shape[2] // np.power(2, num_wt)
    img[:,:,patch_dim:,patch_dim:].fill_(0)
    
    return img

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