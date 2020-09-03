import os
from PIL import Image
import os.path
import io
import string
import random
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.datasets as dset
from torchvision.datasets.lsun import LSUNClass
from torchvision.datasets.utils import verify_str_arg, iterable_to_str
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
from collections.abc import Iterable
import pickle
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

# Pytorch LSUN Dataset with object categories support (cats) added
class LSUN_Objects(VisionDataset):
    """
    `LSUN <https://www.yf.io/p/lsun>`_ dataset.

    Args:
        root (string): Root directory for the database files.
        classes (string or list): One of {'train', 'val', 'test'} or a list of
            categories to load. e,g. ['bedroom_train', 'church_outdoor_train'].
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, classes='train', transform=None, target_transform=None):
        super(LSUN, self).__init__(root, transform=transform,
                                   target_transform=target_transform)
        self.classes = self._verify_classes(classes)

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for c in self.classes:
            self.dbs.append(LSUNClass(
                root=root + '/' + c + '_lmdb',
                transform=transform))

        self.indices = []
        count = 0
        for db in self.dbs:
            count += len(db)
            self.indices.append(count)

        self.length = count

    def _verify_classes(self, classes):
        categories = ['cat']
        dset_opts = ['train', 'val', 'test']

        try:
            verify_str_arg(classes, "classes", dset_opts)
            if classes == 'test':
                classes = [classes]
            else:
                classes = [c + '_' + classes for c in categories]
        except ValueError:
            if not isinstance(classes, Iterable):
                msg = ("Expected type str or Iterable for argument classes, "
                       "but got type {}.")
                raise ValueError(msg.format(type(classes)))

            classes = list(classes)
            msg_fmtstr = ("Expected type str for elements in argument classes, "
                          "but got type {}.")
            for c in classes:
                verify_str_arg(c, custom_msg=msg_fmtstr.format(type(c)))
                c_short = c.split('_')
                category, dset_opt = '_'.join(c_short[:-1]), c_short[-1]

                msg_fmtstr = "Unknown value '{}' for {}. Valid values are {{{}}}."
                msg = msg_fmtstr.format(category, "LSUN class",
                                        iterable_to_str(categories))
                verify_str_arg(category, valid_values=categories, custom_msg=msg)

                msg = msg_fmtstr.format(dset_opt, "postfix", iterable_to_str(dset_opts))
                verify_str_arg(dset_opt, valid_values=dset_opts, custom_msg=msg)

        return classes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target) where target is the index of the target category.
        """
        target = 0
        sub = 0
        for ind in self.indices:
            if index < ind:
                break
            target += 1
            sub = ind

        db = self.dbs[target]
        index = index - sub

        if self.target_transform is not None:
            target = self.target_transform(target)

        img, _ = db[index]
        return img, target


    def __len__(self):
        return self.length

    def extra_repr(self):
        return "Classes: {classes}".format(**self.__dict__)

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