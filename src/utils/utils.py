import torch
import random

# Zeroing out all other patches than the first for WT image: 4D: B * C * H * W
def zero_patches(img):
    zeros = torch.zeros((img.shape[0], img.shape[1], img.shape[2], img.shape[3]))
    patch_dim = img.shape[2] // 2
    zeros[:,:,:patch_dim,:patch_dim] = img[:,:,:patch_dim,:patch_dim]
    
    return zeros

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