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
    tr.CropFromMask(
        crop_elems=['image', 'gt', 'thin', 'void_pixels'],
        relax=RELAX_CROP,
        zero_pad=ZERO_PAD_CROP,
        adaptive_relax=ADAPTIVE_RELAX,
        prefix=''),
    tr.Resize(
        resize_elems=['image', 'gt', 'thin', 'void_pixels'],
        min_size=MIN_SIZE,
        max_size=MAX_SIZE,),
    tr.ComputeImageGradient(elem='image'),
    tr.ExtremePoints(sigma=10, pert=5, elem='gt'),
    tr.GaussianTransform(
        tr_elems=['extreme_points'],
        mask_elem='gt',
        sigma=10,
        tr_name='points'),
    tr.RandomCrop(
        num_thin=NUM_THIN_SAMPLES,
        num_non_thin=NUM_NON_THIN_SAMPLES,
        crop_size=ROI_SIZE,
        prefix='crop_',
        thin_elem='thin',
        crop_elems=['image', 'gt', 'points', 'void_pixels', 'image_grad']),
    tr.MatchROIs(crop_elem='gt', resolution=LR_SIZE),
    tr.FixedResizePoints(
        resolutions={
            'extreme_points': (LR_SIZE, LR_SIZE)},
        mask_elem='gt',
        prefix='lr_'),
    tr.FixedResize(
        resolutions={
            'image' : (LR_SIZE, LR_SIZE),
            'gt'    : (LR_SIZE, LR_SIZE),
            'void_pixels': (LR_SIZE, LR_SIZE)},
        prefix='lr_'),
    tr.GaussianTransform(
        tr_elems=['lr_extreme_points'],
        mask_elem='lr_gt',
        sigma=10,
        tr_name='lr_points'),
    tr.ToImage(
        norm_elem=['crop_points', 'crop_image_grad', 'lr_points']),
    tr.ConcatInputs(
        cat_elems=['lr_image', 'lr_points'],
        cat_name='concat_lr'),
    tr.ConcatInputs(
        cat_elems=['crop_image', 'crop_points'],
        cat_name='concat'),
    tr.ConcatInputs(
        cat_elems=['crop_image', 'crop_image_grad'],
        cat_name='grad'),
    tr.ExtractEdge(mask_elems=['crop_gt']),
    tr.ExtractEdge(mask_elems=['gt']),
    tr.FixedResize(resolutions={'gt_edge' : (LR_SIZE, LR_SIZE)},  prefix='lr_'),
    tr.RemoveElements(
        rm_elems=['points', 'image', 'gt', 'void_pixels', 'thin', 'image_grad']),
    tr.ToTensor(excludes=['rois'])
]


def transforms_normalize(img, target, mean=111.70, std=62.01):
    
    # Threshold
    target = torch.ge(target, 0.5).float() 

    # Rearrange the inputs    
    #target = target.view(5, 1, 512, 512)
    #img = inputs.view(5, 4, 512, 512)
    img = img[:, :3, :, :] # catch only RGB channels
    
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
    
    def __init__(self, ds, transforms=None):
        self.ds = ds        
        self.transforms = transforms
        
    def get_meta(self, idx):
                
        return self.ds[idx]['meta']
    
    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item['concat']
        target = item['crop_gt']
        
        if self.transforms:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def __len__(self):
        return len(self.ds)
    


def create_datasets(dataset_path=DATASET_PATH, split='train', custom_transforms=default_transforms, use_thin=False):
    
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
