import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torchvision.datasets as dset
import os, sys
import numpy as np
from vae_models import WT, wt, IWT, iwt, IWTAE_128_Mask_2, IWTAE_512_Mask_2
from utils.utils import zero_patches, zero_mask, set_seed, save_plot, create_inv_filters, create_filters, zero_pad
import gc
from random import sample
import IPython

if __name__ == "__main__":
    # Import Progressive GAN model (256 x 256)
    use_gpu = True if torch.cuda.is_available() else False

    # trained on high-quality celebrity faces "celebA" dataset
    # this model outputs 512 x 512 pixel images
    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                        'PGAN', model_name='celebAHQ-512',
                        pretrained=True, useGPU=use_gpu)
    # this model outputs 256 x 256 pixel images
    # model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
    #                        'PGAN', model_name='celebAHQ-256',
    #                        pretrained=True, useGPU=1)

    DEVICE = 'cuda:0'
    checkpoint_128_path = '/disk_c/han/wtvae_results/models/iwtae128_noimgloss_lr1e-4_nonorm_2iwt_full/iwtvae_epoch300.pth'
    checkpoint_512_path = '/disk_c/han/wtvae_results/models/iwtae512_overfit_noimgloss_lr1e-4_nonorm_full_rerun/iwtvae_epoch272.pth'
    dataroot = '/disk_c/han/data/celeba_512/'
    IMAGE_SIZE = 512
    BATCH_SIZE = 16
    Z_DIM = 100
    NUM_WT = 4
    NUM_IWT = 2
    TARGET_DIM = 512
    NUM_EPOCHS = 200
    img_output_dir = '/disk_c/han/wtvae_results/image_samples/prog_gan_128_512_hierarchical_eval_4/'

    try:
        os.mkdir(img_output_dir)
    except:
        raise Exception('Cannot create image output directory')

    set_seed(2020)

    # Create the dataset and sample 10,000 images from each
    celeba_ds = dset.ImageFolder(root=dataroot,
                                transform=transforms.Compose([
                                transforms.Resize(IMAGE_SIZE),
                                transforms.CenterCrop(IMAGE_SIZE),
                                transforms.ToTensor(),
                            ]))

    sample_10000_idx = sample(range(len(celeba_ds)), 10000)

    celeba_ds = torch.utils.data.Subset(celeba_ds, sample_10000_idx)

    # Create the dataloader 
    dataloader = torch.utils.data.DataLoader(celeba_ds,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=4)