import torch

# Zeroing out all other patches than the first for WT image: 4D: B * C * H * W
def zero_patches(img):
    zeros = torch.zeros((img.shape[0], img.shape[1], img.shape[2], img.shape[3]))
    patch_dim = img.shape[2] // 2
    zeros[:,:,:patch_dim,:patch_dim] = img[:,:,:patch_dim,:patch_dim]
    
    return zeros