""" Train TOS-Net. """
import os
os.environ['OMP_NUM_THREADS'] = "8"
from collections import OrderedDict
from datetime import datetime
import glob
import numpy as np
import argparse
import random
import json

# PyTorch includes
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate, sigmoid
import torch.backends.cudnn as cudnn

# Custom includes
import dataloaders.thinobject5k as thinobject5k
import dataloaders.custom_transforms as tr
import dataloaders.helpers as helpers


from torch.utils.data import Dataset, DataLoader

# Default settings
EXPERIMENT_NAME = 'ThinObject5K_for_RESNET_FPN'
RANDOM_SEED = 1234
# Network-specific arguments
LR_SIZE = 512                   # size of context stream
# Training-specific arguments
NUM_THIN_SAMPLES = 4            # number of samples consisting of thin parts
NUM_NON_THIN_SAMPLES = 1        # number of samples consisting of non-thin parts
MIN_SIZE = 512                  # minimum image size allowed
MAX_SIZE = 1980                 # maximum image size allowed
ROI_SIZE = 512                  # patch size for training
BATCH_SIZE = 1                  # batch size for training
NUM_WORKERS = 6                 # number of workers to read daaset
RELAX_CROP = 50                 # enlarge bbox by 'relax_crop' pixels
ZERO_PAD_CROP = True            # insert zero padding when cropping
ADAPTIVE_RELAX = True           # compute 'relax_crop' adaptively?
DATASET = ['ThinObject5k_OnlyThinRegions']      # dataset for training
DATASET_PATH = '../../datasets/ThinObject5k_OnlyThinRegions'
SAVE_DIR = './experiments'



    
# Setup data transformations
default_transforms = [
    tr.RandomHorizontalFlip(),
    # Crop region where ground truth is 1
    tr.CropFromMask(
        crop_elems=['image', 'gt', 'thin', 'void_pixels'],
        relax=RELAX_CROP,
        zero_pad=ZERO_PAD_CROP,
        adaptive_relax=ADAPTIVE_RELAX,
        prefix=''),
    # Resize images, min_size = 512, max_size = 1980
    tr.Resize(
        resize_elems=['image', 'gt', 'thin', 'void_pixels'],
        min_size=MIN_SIZE,
        max_size=MAX_SIZE),
    # Selects 4 random points where thin==1 and 1 random point from a sample,
    # and crops 5 patchs whit size 512x512 around these points
    # adds prefi 'crop_' to the keys of the new samples.  
    tr.RandomCrop(
        num_thin=NUM_THIN_SAMPLES,
        num_non_thin=NUM_NON_THIN_SAMPLES,
        crop_size=ROI_SIZE,
        prefix='crop_',
        thin_elem='thin',
        crop_elems=['image', 'gt', 'void_pixels']),
    tr.ToTensor(excludes=['rois'])
]

def collate_fn(samples):
    """
    Collates a list of samples into a batch.

    Parameters:
        samples (list): A list of samples, where each sample is a tuple of (image, target).

    Returns:
        tuple: A tuple containing the batched images and targets.
    """
    imgs, targets = zip(*samples)
    imgs = torch.cat(imgs)
    targets = torch.cat(targets)
    
    return imgs, targets


def transforms_normalize(img, target, mean=111.70, std=62.01):
    
    # Threshold
    target = torch.ge(target, 0.5).long() 
    # img = img[:, :3, :, :] # catch only RGB channels
    
    # Normalize by mean and std
    img = (img - mean) / std    
    
    return img, target

def subset(dataset_path=DATASET_PATH, split='train', custom_transforms=default_transforms, use_thin=False):
    
    composed_transforms_tr = transforms.Compose(custom_transforms)
    
    """Return the subset of the dataset."""
    ds = thinobject5k.ThinObject5K(
        root=dataset_path,
        split=split, 
        transform=composed_transforms_tr, 
        use_thin=use_thin
    )
    p = OrderedDict()
    p['dataset_train'] = str(ds)
    p['transformations_train'] = [str(tran) for tran in composed_transforms_tr.transforms]

    helpers.generate_param_report(os.path.join(SAVE_DIR, EXPERIMENT_NAME + '.txt'), p)
    
    return ds

class ThinObjectDatasetWrapper(Dataset):
    """
    A wrapper class for a thin object dataset.

    This class wraps an existing dataset and provides additional functionality
    such as applying transformations to the data and extracting metadata.

    Parameters:
        ds (Dataset): The underlying dataset to wrap.
        transforms (callable, optional): A function or transform to apply to the data.

    Attributes:
        ds (Dataset): The underlying dataset.
        transforms (callable): The transformation function or transform to apply.

    """

    def __init__(self, ds, transforms=None):
        self.ds = ds        
        self.transforms = transforms   
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Parameters:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and target.

        """
        item = self.ds[idx]
        img = item['crop_image']
        target = item['crop_gt']
        
        if self.transforms:
            img, target = self.transforms(img, target)
        
        return img, target[:, 0] # return only the first channel of the target
    
    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return len(self.ds)
    
    def get_meta(self, idx):
        """
        Get the metadata of an item in the dataset.

        Parameters:
            idx (int): The index of the item.

        Returns:
            dict: The metadata of the item.

        """
        return self.ds[idx]['meta']
    


def create_datasets(dataset_path=DATASET_PATH, split='train', custom_transforms=default_transforms, use_thin=False):
    """
    Create datasets for training, validation, or testing.

    Parameters:
        dataset_path (str): Path to the dataset.
        split (str): Split of the dataset to create ('train', 'val', or 'test').
        custom_transforms (list): List of custom transforms to apply to the dataset.
        use_thin (bool): Flag indicating whether to use thin objects in the dataset.

    Returns:
        If split is 'train':
            ds_train (Dataset): Training dataset.
            ds_valid (Dataset): Validation dataset.
        If split is 'val' or 'test':
            ds (Dataset): Validation or testing dataset.
    """
    
    if split == 'train':
        ds_train = subset(dataset_path=dataset_path, split='train', custom_transforms=custom_transforms, use_thin=use_thin)
        ds_valid = subset(dataset_path=dataset_path, split='val', custom_transforms=custom_transforms, use_thin=use_thin)
        
        ds_train = ThinObjectDatasetWrapper(ds=ds_train, transforms=transforms_normalize)
        ds_valid = ThinObjectDatasetWrapper(ds=ds_valid, transforms=transforms_normalize)        
        
        return ds_train, ds_valid
    
    if split == 'val':
        ds = subset(dataset_path=dataset_path, split=split, custom_transforms=custom_transforms, use_thin=use_thin)
        ds = ThinObjectDatasetWrapper(ds=ds, transforms=transforms_normalize)  
        return ds
    
    if split == 'test':                
        ds = subset(dataset_path=dataset_path, split=split, custom_transforms=custom_transforms, use_thin=use_thin)
        ds = ThinObjectDatasetWrapper(ds=ds, transforms=transforms_normalize)           
        return ds
