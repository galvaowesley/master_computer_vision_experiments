'''Creates the vessel_mini dataset.'''

from functools import partial
import numpy as np
from PIL import Image
import torch
import albumentations as aug
from albumentations.pytorch import ToTensorV2
from torchtrainer.imagedataset import ImageSegmentationDataset
from torchvision.datasets.utils import download_and_extract_archive

def download(directory):

    url = 'https://www.dropbox.com/s/2a487667dg6266e/vessel_crop.tar.gz?dl=1'
    download_root = directory
    extract_root = directory
    filename = 'vessel_crop.tar.gz'
    download_and_extract_archive(url, download_root, extract_root, filename, remove_finished=True)
    
def cat_list(images, fill_value=0): 
    """
    Concatenates a list of images into a batch tensor.

    Parameters
    ----------
        images (list): A list of images to be concatenated.
        fill_value (int, optional): The value used to fill the padded regions. Defaults to 0.

    Returns
    -------
        torch.Tensor: The batch tensor containing the concatenated images.
    """
    
    # Determine the max dims for batch
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        
    if len(max_size) == 2:
        # Pad to the max dims
        max_size = ( max_size[0]+32 - max_size[0]%32, max_size[1]+32 - max_size[1]%32)
    
    if len(max_size) == 3:
        # Pad to the max dims
        max_size = (max_size[0], max_size[1]+32 - max_size[1]%32, max_size[2]+32 - max_size[2]%32)

    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)   
    
    for idx, (img, pad_img) in enumerate(zip(images, batched_imgs)):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
        
    
    return batched_imgs

# def collate_fn(batch):
#     '''
#     Function for collating batch.

#     Parameters:
#         batch (list): A list of tuples containing images and targets.

#     Returns:
#         tuple: A tuple containing the batched images and targets.
#     '''
#     images, targets = list(zip(*batch))
#     batched_imgs = cat_list(images, fill_value=0)
#     batched_targets = cat_list(targets, fill_value=255)
#     return batched_imgs[0], batched_targets[0, :, 0]

def collate_fn_cat(samples):
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

def collate_fn_stack(samples):
    """
    Collates a list of samples into a batch. In this case, the images are stacked instead of concatenated.

    Parameters:
        samples (list): A list of samples, where each sample is a tuple of (image, target).

    Returns:
        tuple: A tuple containing the batched images and targets.
    """
    imgs, targets = zip(*samples)
    imgs = torch.stack(imgs)
    targets = torch.stack(targets)
    
    return imgs, targets

def name_to_label_map(img_path):
    return img_path.replace('.tiff', '.png')

def img_opener(img_path):
    img_pil = Image.open(img_path)
    return np.array(img_pil)

def label_opener(img_path):
    img_pil = Image.open(img_path)
    return np.array(img_pil)//255

def transform(image, mask, transform_comp):
    """Given an image and a mask, apply transforms in `transform_comp`"""
    res = transform_comp(image=image, mask=mask)
    image, mask = res['image'], res['mask']

    image = torch.cat([image] * 3, dim=0) # Repeat the image 3 times to simulate RGB channels
    
    return image, mask.long()

def zscore(img, **kwargs):
    return ((img-img.mean())/img.std()).astype(np.float32)

def create_transform(mean, std, crop_size, type='train'):
    """Create a transform function with signature transform(image, label)."""

    if crop_size is None:
        crop_transf = aug.NoOp()
    else:
        crop_transf = aug.RandomCrop(crop_size[0], crop_size[1])
    if type=='train':
        transform_comp = aug.Compose([
            crop_transf,
            aug.CLAHE(clip_limit=(3., 3.), tile_grid_size=(16, 16), p=1.),
            aug.Normalize(mean=mean, std=std),
            aug.Lambda(name='zscore', image=zscore, p=1.),
            ToTensorV2()
        ])
    elif type=='validate':
        transform_comp = aug.Compose([
            # aug.CenterCrop(256, 256),   # Still need to crop for validation because some samples have different sizes, which complicates batch creation
            aug.RandomCrop(crop_size[0], crop_size[1]),
            aug.CLAHE(clip_limit=(3., 3.), tile_grid_size=(16, 16), p=1.),
            aug.Normalize(mean=mean, std=std),
            aug.Lambda(name='zscore', image=zscore, p=1.),
            ToTensorV2()
        ])        

    transform_func = partial(transform, transform_comp=transform_comp)

    return transform_func

def create_datasets(img_dir, label_dir, crop_size=(256,256), train_val_split=0.1, seed=None):
    """Create dataset from directory. 
    
    Args
    crop_size: (height,width) to crop the images
    train_val_split: percentage of images used for validation
    use_simple: if True, use only crop, normalization and ToTensor. If False, use many data augmentation
    transformations.
    seed: seed for splitting the data
    """

    mean_data = 0.
    std_data = 1.

    train_transform = create_transform(mean_data, std_data, crop_size, type='train')
    valid_transform = create_transform(mean_data, std_data, crop_size, type='validate')

    ds = ImageSegmentationDataset(img_dir, label_dir, name_to_label_map, img_opener=img_opener, label_opener=label_opener)
    ds_train, ds_valid = ds.split_train_val(train_val_split, seed=seed)
    ds_train.set_transform(train_transform)
    ds_valid.set_transform(valid_transform)

    return ds_train, ds_valid


def get_statistics(ds):
    """
    Get the mean and standard deviation of the dataset. Besides, it also calculates 
    the percentage of object and background pixels.
    
    Parameters
    ----------
        ds (Dataset): The dataset for which to calculate the statistics.
        
    Returns
    -------
        tuple: A tuple containing the mean and standard deviation of the dataset.
    """

    n = len(ds)
    mean = 0.
    std = 0.
    
    count_pixels_obj = 0 # Object pixels
    count_pixels_bg = 0 # Background pixels
    
    for img, target in ds:
        mean += np.mean(np.array(img))
        std += np.std(np.array(img))
        
        count_pixels_bg += np.sum(np.array(target) == 0)
        count_pixels_obj += np.sum(np.array(target) == 1)
    
        
    mean /= n
    std /= n
    
    percent_obj = count_pixels_obj / (count_pixels_obj + count_pixels_bg)
    percent_bg = count_pixels_bg / (count_pixels_obj + count_pixels_bg)
    
    print(f'Mean: {mean:.2f}')
    print(f'Std: {std:.2f}')
    
    print(f'Percentage of object pixels: {percent_obj:.2f}')
    print(f'Percentage of background pixels: {percent_bg:.2f}')
    
    
    #return mean, std