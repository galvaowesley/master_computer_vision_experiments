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
from layers.loss import binary_cross_entropy_loss, dice_loss, bootstrapped_cross_entopy_loss
import networks.tosnet as tosnet

# Default settings
MODEL_NAME = 'TOSNetFineTuning'
RANDOM_SEED = 1234
# Network-specific arguments
NUM_INPUTS = 4                  # input channels
NUM_CLASSES = 1                 # number of classes
BACKBONE = 'resnet50'           # backbone architecture
LR_SIZE = 512                   # size of context stream
# Training-specific arguments
NUM_THIN_SAMPLES = 4            # number of samples consisting of thin parts
NUM_NON_THIN_SAMPLES = 1        # number of samples consisting of non-thin parts
MIN_SIZE = 512                  # minimum image size allowed
MAX_SIZE = 1980                 # maximum image size allowed
ROI_SIZE = 512                  # patch size for training
NUM_EPOCHS = 50                 # number of epochs for training
BATCH_SIZE = 1                  # batch size for training
SNAPSHOT = 10                   # store a model every 'snapshot'
LEARNING_RATE = 1e-3            # learning rate for training
WEIGHT_DECAY = 0.0005           # weight decay for training
MOMENTUM = 0.9                  # momentum for training
NUM_WORKERS = 6                 # number of workers to read daaset
RELAX_CROP = 50                 # enlarge bbox by 'relax_crop' pixels
ZERO_PAD_CROP = True            # insert zero padding when cropping
ADAPTIVE_RELAX = True           # compute 'relax_crop' adaptively?
DISPLAY = 1                    # print stats every 'display' iterations
CONTEXT_LOSS = {'bbce': 1}                      # losses for training context branch
MASK_LOSS = {'bootstrapped_ce': 1, 'dice': 1}   # losses for training mask prediction
EDGE_LOSS = {'bbce': 1, 'dice': 1}              # losses for training hr edge branch
DATASET = ['thinobject5k']      # dataset for training
LOSS_AVERAGE = 'size'           # how to average the loss
LR_SCHEDULE = 'poly'            # learning rate scheduler
BOOTSTRAPPED_RATIO = 1./16      # multiplier for determining #pixels in bootstrapping



    
# Setup data transformations
composed_transforms = [
    tr.RandomHorizontalFlip(),
    tr.CropFromMask(
        crop_elems=['image', 'gt', 'thin', 'void_pixels'],
        relax=args.relax_crop,
        zero_pad=args.zero_pad_crop,
        adaptive_relax=args.adaptive_relax,
        prefix=''),
    tr.Resize(
        resize_elems=['image', 'gt', 'thin', 'void_pixels'],
        min_size=args.min_size,
        max_size=args.max_size),
    tr.ComputeImageGradient(elem='image'),
    tr.ExtremePoints(sigma=10, pert=5, elem='gt'),
    tr.GaussianTransform(
        tr_elems=['extreme_points'],
        mask_elem='gt',
        sigma=10,
        tr_name='points'),
    tr.RandomCrop(
        num_thin=args.num_thin_samples,
        num_non_thin=args.num_non_thin_samples,
        crop_size=args.roi_size,
        prefix='crop_',
        thin_elem='thin',
        crop_elems=['image', 'gt', 'points', 'void_pixels', 'image_grad']),
    tr.MatchROIs(crop_elem='gt', resolution=args.lr_size),
    tr.FixedResizePoints(
        resolutions={
            'extreme_points': (args.lr_size, args.lr_size)},
        mask_elem='gt',
        prefix='lr_'),
    tr.FixedResize(
        resolutions={
            'image' : (args.lr_size, args.lr_size),
            'gt'    : (args.lr_size, args.lr_size),
            'void_pixels': (args.lr_size, args.lr_size)},
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
    tr.RemoveElements(
        rm_elems=['points', 'image', 'gt', 'void_pixels', 'thin', 'image_grad']),
    tr.ToTensor(excludes=['rois'])]
composed_transforms_tr = transforms.Compose(composed_transforms)
    


# Setup dataset
db_train = thinobject5k.ThinObject5K(
    root=args.dataset_path,
    split='train', 
    transform=composed_transforms_tr, 
    use_thin=False
)


p['dataset_train'] = str(db_train)
p['transformations_train'] = [str(tran) for tran in composed_transforms_tr.transforms]

trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, \
                        num_workers=args.num_workers, drop_last=True)
helpers.generate_param_report(os.path.join(save_dir, args.model_name + '.txt'), p)

# Train variables
num_img_tr = len(trainloader)
num_patch = args.num_thin_samples + args.num_non_thin_samples
num_batch = num_patch * args.batch_size
print('Training network')
pretrained_net.train()
    
# Main training loop
for epoch in range(args.num_epochs):
    for ii, sample in enumerate(trainloader):

        ### Uncomment to visualize ###
        # _visualize_minibatch(sample, args)

        # Read inputs and groundtruths
        inputs      = sample['concat'].to(device)
        inputs_lr   = sample['concat_lr'].to(device)
        grads       = sample['grad'].to(device)
        voids       = sample['crop_void_pixels'].to(device) # NEW
        voids_lr    = sample['lr_void_pixels'].to(device) # NEW
        gts         = sample['crop_gt'].to(device)
        gts_lr      = sample['lr_gt'].to(device)
        gts_edge    = sample['crop_gt_edge'].to(device)
        # Threshold
        gts = torch.ge(gts, 0.5).float() 
        gts_lr = torch.ge(gts_lr, 0.5).float()
        gts_edge = torch.ge(gts_edge, 0.5).float()
        
        # Read rois and rearrange the inputs
        rois = sample['rois'].float().view(-1, 4)
        batch_ind = torch.arange(args.batch_size).float().unsqueeze(1).repeat(1, 
                        num_patch).view(-1, 1) # attach batch id
        rois = torch.cat((batch_ind, rois), dim=1).to(device)
        inputs = inputs.view(num_batch, 4, args.roi_size, args.roi_size)
        grads = grads.view(num_batch, 4, args.roi_size, args.roi_size)
        gts = gts.view(num_batch, 1, args.roi_size, args.roi_size)
        gts_edge = gts_edge.view(num_batch, 1, args.roi_size, args.roi_size)
        voids = voids.view(num_batch, 1, args.roi_size, args.roi_size)

        # Forward pass
        outs = pretrained_net.forward(inputs, grads, inputs_lr, rois)
        outs_lr, outs, edges = outs
        outs_lr = interpolate(outs_lr, gts_lr.size()[2:], mode='bilinear',
                        align_corners=True)
        
