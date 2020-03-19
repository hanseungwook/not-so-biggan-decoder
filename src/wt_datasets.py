import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
# from scipy.misc import imresize

# Celeba Dataset
# If WT == False, returns original image
# Else WT == True, returns (original image, WT image)
class CelebaDataset(Dataset):

    def __init__(self, root_dir, img_list, WT=False):
        self.root_dir = root_dir
        self.img_list = img_list
        self.WT = WT

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_dir, self.img_list[idx]))
        img = np.array(img)
        img = img / 255
        img = torch.from_numpy(img.transpose(2,0,1)).float()
        
        # Returning both original image and WT image if self.WT
        if self.WT:
            img_wt = wt(img_wt.unsqueeze(1)).squeeze()

            return img, img_wt
        
        return img

# CelebaDataset that returns a pair of same images in different dimensions
class CelebaDatasetPair(Dataset):

    def __init__(self, root_dir1, root_dir2, img_list, WT=False):
        self.root_dir1 = root_dir1
        self.root_dir2 = root_dir2
        self.img_list = img_list
        self.WT = WT

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img1 = Image.open(os.path.join(self.root_dir1, self.img_list[idx]))
        img1 = np.array(img1)
        img1 = img1 / 255
        img1 = torch.from_numpy(img1.transpose(2,0,1)).float()
        
        # Returning both original image and WT image if self.WT
        # if self.WT:
        #     img_wt = wt(img_wt.unsqueeze(1)).squeeze()

        #     return img, img_wt
        img2 = Image.open(os.path.join(self.root_dir2, self.img_list[idx]))
        img2 = np.array(img2)
        img2 = img2 / 255
        img2 = torch.from_numpy(img2.transpose(2,0,1)).float()
        
        return (img1, img2)
    
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors. numpy image: H x W x C, torch image: C X H X W
    """

    def __call__(self, image, invert_arrays=True):

        if invert_arrays:
            image = image.transpose((2, 0, 1))[:,:h_img*2,:w_img*2]

        return torch.from_numpy(image)