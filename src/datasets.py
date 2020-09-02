import os
from PIL import Image
import random
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from wt_utils import get_3masks

####################################################################
# DATASET HELPER
####################################################################

def parse_dataset_args(dataset):
    dataset = dataset.split('-')
    ds_name = None
    classes = None
    
    if 'imagenet' in dataset:
        ds_name = 'imagenet'
    elif 'lsun' in dataset:
        ds_name = 'lsun'
        classes = dataset[1:]
        classes = [[c + ds_type for c in classes] for ds_type in ['_train', '_val']]
    else:
        raise NotImplementedError()

    return ds_name, classes

def create_dataset(ds_name, path, transform, classes=None):
    dataset = None
    
    if ds_name == 'imagenet':
        dataset = dset.ImageFolder(root=path, transform=transform)
    elif ds_name == 'lsun':
        dataset = dset.LSUN(path, classes=classes, transform=transform)
    else:
        raise NotImplementedError()
    
    return dataset


####################################################################
# CUSTOM DATASET OBJECTS
####################################################################

# Vanilla Imagenet dataset that replicates ImageFolder
class ImagenetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_list = os.listdir(root_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_dir, self.img_list[idx]))
        img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)
        
        return img


# Imagenet dataset that returns both (data augmented) image and respective mask at respective level
class ImagenetDataAugDataset(Dataset):
    def __init__(self, root_dir, num_wt, mask_dim, wt, filters, default_transform=None, transform=None, p=0.5):
        self.root_dir = root_dir
        self.img_list = os.listdir(root_dir)

        self.num_wt = num_wt
        self.mask_dim = mask_dim
        self.wt = wt
        self.filters = filters
        
        self.default_transform = default_transform
        self.transform = transform
        self.p = p
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_dir, self.img_list[idx]))
        img = img.convert('RGB')
        img_t = img.copy()
        mask = None
        t = None
            
        # Select 1 random transform and apply (these accept PIL)
        if self.p < random.random() and self.transform:
            t = random.choice(self.transform)
            if not isinstance(t, transforms.RandomErasing):
                img_t = t(img)
        
        # Apply default transform
        if self.default_transform:
            img = self.default_transform(img)
        
        # If selected transform is random erasing, apply after default transform (accepts tensor)
        if isinstance(t, transforms.RandomErasing):
            img_t = t(img)
        # If selected transform is not random erasing or none, just apply default transform
        else:
            img_t = self.default_transform(img_t)
        
        # If transform is color jitter, get mask of original
        if isinstance(t, torchvision.transforms.ColorJitter):
            mask = self.wt(img.unsqueeze(0), self.filters, levels=self.num_wt)[:, :, :self.mask_dim*2, :self.mask_dim*2]
        else:
            mask = self.wt(img_t.unsqueeze(0) , self.filters, levels=self.num_wt)[:, :, :self.mask_dim*2, :self.mask_dim*2]
        
        masks = get_3masks(mask, self.mask_dim)
        
        return img_t, masks
    

class SampleDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.data = np.load(file_path)['x']
        self.transform = transform
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx])

        if self.transform:
            img = self.transform(img)
        
        return img