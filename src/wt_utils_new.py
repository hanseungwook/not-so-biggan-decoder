import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import random
import pywt
import IPython
#from logger import Logger

################# ZERO FUNCTIONS #################

# Zeroing out all other patches than the first for WT image: 4D: B * C * H * W
def zero_patches(img, num_wt):
    padded = torch.zeros(img.shape, device=img.device)
    patch_dim = img.shape[2] // np.power(2, num_wt)
    padded[:, :, :patch_dim, :patch_dim] = img[:, :, :patch_dim, :patch_dim]
    
    return padded


# Create padding on patch so that this patch is formed into a square image with other patches as 0
# 3 x 128 x 128 => 3 x target_dim x target_dim
def zero_pad(img, target_dim, device='cpu'):
    batch_size = img.shape[0]
    num_channels = img.shape[1]
    padded_img = torch.zeros((batch_size, num_channels, target_dim, target_dim), device=device)
    padded_img[:, :, :img.shape[2], :img.shape[3]] = img
    
    return padded_img
    

# Zeroing out the first patch's portion of the mask
def zero_mask(mask, num_iwt, cur_iwt):
    padded = torch.zeros(mask.shape, device=mask.device)
    h = mask.shape[2]

    inner_patch_h0 = h // (np.power(2, num_iwt-cur_iwt+1))
    inner_patch_w0 = h // (np.power(2, num_iwt-cur_iwt+1))

    if len(mask.shape) == 3:
        padded[:, inner_patch_h0:, :] = mask[:, inner_patch_h0:, :]
        padded[:, :inner_patch_h0, inner_patch_w0:] = mask[:, :inner_patch_h0, inner_patch_w0:]
    elif len(mask.shape) == 4:
        padded[:, :, inner_patch_h0:, :] = mask[:, :, inner_patch_h0:, :]
        padded[:, :, :inner_patch_h0, inner_patch_w0:] = mask[:, :, :inner_patch_h0, inner_patch_w0:]
    
    return padded


################# WT FUNCTIONS #################
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

def wt_successive(vimg, filters, levels=1):
    bs = vimg.shape[0]
    h = vimg.size(2)
    w = vimg.size(3)
    vimg = vimg.reshape(-1, 1, h, w)
    padded = torch.nn.functional.pad(vimg,(2,2,2,2))
    res = torch.nn.functional.conv2d(padded, Variable(filters[:,None]),stride=2)

    cnt = levels-1
    while(cnt>0):
        h = h//2
        w = w//2
        res = res.reshape(-1, 1, h, w) #batch*3*4
        padded = torch.nn.functional.pad(res,(2,2,2,2))
        res = torch.nn.functional.conv2d(padded, Variable(filters[:,None]), stride=2)
        cnt-=1
    res = res.reshape(bs, -1, h//2, w//2)

    return res

# Returns IWT of img with only TL patch (low frequency) zero-ed out
def wt_hf(vimg, filters, levels=1):
    # Apply WT
    wt_img = wt(vimg, filters, levels)

    # Zero out TL patch
    wt_img_hf = zero_mask(wt_img, levels, 1)

    # Apply IWT
    inv_filters = create_inv_filters(wt_img_hf.device)
    iwt_img_hf = iwt(wt_img_hf, inv_filters, levels)

    return iwt_img_hf

# Returns IWT of img with only TL patch (low-frequency) -- high frequencies all zero-ed out
def wt_lf(vimg, filters, levels=1):
    # Apply WT
    wt_img = wt(vimg, filters, levels)
    h = wt_img.shape[2]
    w = wt_img.shape[3]

    assert h == w

    # Zero out everything other than TL patch
    wt_img_padded = zero_pad(wt_img[:, :, :h // (2**levels), :w // (2 ** levels)], h, device=wt_img.device)

    # Apply IWT
    inv_filters = create_inv_filters(wt_img_padded.device)
    iwt_img_lf = iwt(wt_img_padded, inv_filters, levels)

    return iwt_img_lf
    

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


# Input is 256 x 256 or 128 x 128 (levels automatically adjusted), and outputs 128 x 128 will all patches WT'ed to 32 x 32
def wt_128_3quads(img, filters, levels):
    data = img.clone()
        
    data = wt(data, filters, levels)[:, :, :128, :128]
    h = data.shape[2]
    w = data.shape[3]
    
    tr = wt(data[:, :, :h//2, w//2:], filters, levels=1)
    bl = wt(data[:, :, h//2:, :w//2], filters, levels=1)
    br = wt(data[:, :, h//2:, w//2:], filters, levels=1)
    
    data[:, :, :h//2, w//2:] = tr
    data[:, :, h//2:, :w//2] = bl
    data[:, :, h//2:, w//2:] = br
    
    return data


def wt_256_3quads(data, filters, levels):
    h = data.shape[2]
    w = data.shape[3]
    
    data = wt(data, filters, levels)[:, :, :256, :256]

    # Applying WT to 64x64 inner 3 quadrants
    data[:, :, :64, 64:128] = wt(data[:, :, :64, 64:128], filters, levels=1)
    data[:, :, 64:128, :64] = wt(data[:, :, 64:128, :64], filters, levels=1)
    data[:, :, 64:128, 64:128] = wt(data[:, :, 64:128, 64:128], filters, levels=1)
    
    data[:, :, :h//2, w//2:] = wt_128_3quads(data[:, :, :h//2, w//2:], filters, levels=2)
    data[:, :, h//2:, :w//2] = wt_128_3quads(data[:, :, h//2:, :w//2], filters, levels=2)
    data[:, :, h//2:, w//2:] = wt_128_3quads(data[:, :, h//2:, w//2:], filters, levels=2)
    
    return data

def apply_iwt_quads_128(img_quad, inv_filters):
    h = img_quad.shape[2] // 2
    w = img_quad.shape[3] // 2
    
    img_quad[:, :, :h, w:] = iwt(img_quad[:, :, :h, w:], inv_filters, levels=1)
    img_quad[:, :, h:, :w] = iwt(img_quad[:, :, h:, :w], inv_filters, levels=1)
    img_quad[:, :, h:, w:] = iwt(img_quad[:, :, h:, w:], inv_filters, levels=1)
    
    img_quad = iwt(img_quad, inv_filters, levels=2)
    
    return img_quad

################# COLLATE/SPLIT FUNCTIONS #################

# Gets 3 masks in order of top-right, bottom-left, and bottom-right quadrants
def get_3masks(img, mask_dim):
    tr = img[:, :, :mask_dim, mask_dim:]
    bl = img[:, :, mask_dim:, :mask_dim]
    br = img[:, :, mask_dim:, mask_dim:]
    
    return tr.squeeze(), bl.squeeze(), br.squeeze()


# Gets 4 masks in order of top-left, top-right, bottom-left, and bottom-right quadrants
def get_4masks(img, mask_dim):
    tl = img[:, :, :mask_dim, :mask_dim]
    tr = img[:, :, :mask_dim, mask_dim:]
    bl = img[:, :, mask_dim:, :mask_dim]
    br = img[:, :, mask_dim:, mask_dim:]
    
    return tl, tr, bl, br
    

# Receives WT'ed image containing 4 equal-sized quadrants with 3 masks to be collated
def collate_masks_channels(img):
    tr, bl, br = get_masks(img)
    
    return torch.cat((tr, bl, br), dim=1)


# Splits 3 masks collated channel-wise
def split_masks_from_channels(data):
    nc_mask = data.shape[1] // 3
    
    return data[:, :nc_mask, :, :], data[:, nc_mask:2*nc_mask, :, :], data[:, 2*nc_mask:, :, :]


# Splits four patches from a square/grid
def create_patches_from_grid(data):
    h = data.shape[2]
    w = data.shape[3]
    assert(h == w)
    
    tl = data[:, :, :h//2, :w//2]    
    tr = data[:, :, :h//2, w//2:]
    bl = data[:, :, h//2:, :w//2]
    br = data[:, :, h//2:, w//2:]
    
    return torch.stack((tl, tr, bl, br), dim=1)


# From a square/grid, collate into channels (4 patches)
def collate_channels_from_grid(data):
    h = data.shape[2]
    w = data.shape[3]
    assert(h == w)
    
    tl = data[:, :, :h//2, :w//2]    
    tr = data[:, :, :h//2, w//2:]
    bl = data[:, :, h//2:, :w//2]
    br = data[:, :, h//2:, w//2:]
    
    return torch.cat((tl, tr, bl, br), dim=1)

# Splits 16 patches from a square/grid
def create_patches_from_grid_16(data):
    h = data.shape[2]
    w = data.shape[3]
    assert(h == w)
    
    tl = data[:, :, :h//2, :w//2]
    tl_patches = create_patches_from_grid(tl)
    tr = data[:, :, :h//2, w//2:]
    tr_patches = create_patches_from_grid(tr)
    bl = data[:, :, h//2:, :w//2]
    bl_patches = create_patches_from_grid(bl)
    br = data[:, :, h//2:, w//2:]
    br_patches = create_patches_from_grid(br)

    return torch.cat((tl_patches, tr_patches, bl_patches, br_patches), dim=1)


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


# Assumes four patches concatenated channel-wise and converts into image
def collate_channels_to_img(img_channels, device='cpu'):
    bs = img_channels.shape[0]
    c = img_channels.shape[1] // 4
    h = img_channels.shape[2]
    w = img_channels.shape[3]
    
    img = collate_patches_to_img(img_channels[:,:c], img_channels[:,c:2*c], img_channels[:, 2*c:3*c], img_channels[:, 3*c:], device)
    
    return img


# Assumes 16 patches concatenated channel-wise and converts into image
def collate_16_channels_to_img(img_channels, device='cpu'):
    bs = img_channels.shape[0]
    c = img_channels.shape[1] // 4
    h = img_channels.shape[2]
    w = img_channels.shape[3]
    
    tl = collate_channels_to_img(img_channels[:, :c], device)
    tr = collate_channels_to_img(img_channels[:, c:2*c], device)
    bl = collate_channels_to_img(img_channels[:, 2*c:3*c], device)
    br = collate_channels_to_img(img_channels[:, 3*c:], device)
    
    img = collate_patches_to_img(tl, tr, bl, br, device)
    
    return img

################# MISC #################

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

def calc_grad_norm_2(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    return total_norm


def load_checkpoint(model, optimizer, model_type, args):
    checkpoint = torch.load(args.output_dir + '/iwt_model_{}_itr{}.pth'.format(model_type, args.checkpoint), map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    logger = torch.load(args.output_dir + '/logger.pth')

    del checkpoint
    torch.cuda.empty_cache()

    return model, optimizer, logger

def load_weights(model, checkpoint_path, args):
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    del checkpoint
    torch.cuda.empty_cache()

    return model

def freeze_model(model):
    for params in model.parameters():
        params.requires_grad = False
