'''Creates the vessel_mini dataset.'''

from functools import partial
import numpy as np
from PIL import Image
import torch
import albumentations as aug
from albumentations.pytorch import ToTensorV2
import torchtrainer
from torchtrainer.imagedataset import ImageSegmentationDataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import v2

def download(directory):

    url = 'https://www.dropbox.com/s/2a487667dg6266e/vessel_crop.tar.gz?dl=1'
    download_root = directory
    extract_root = directory
    filename = 'vessel_crop.tar.gz'
    download_and_extract_archive(url, download_root, extract_root, filename, remove_finished=True)

def name_to_label_map(img_path):
    return img_path.replace('.jpg', '.png')

def img_opener(img_path):
    img_pil = Image.open(img_path)
    img = np.array(img_pil)
    img = torch.from_numpy(img).permute(2, 0, 1)
    

    return img

def label_opener(img_path):
    label_pil = Image.open(img_path)
    label = np.array(label_pil)//255
    label = torch.from_numpy(label).to(torch.long)

    return label


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
    
    
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs

def collate_fn(batch):
    '''Function for collating batch.'''
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets

def transform(image, mask, transform_comp):
    """Given an image and a mask, apply transforms in `transform_comp`"""
    res = transform_comp(image=image, mask=mask)
    image, mask = res['image'], res['mask']

    return image, mask.long()

def zscore(img, **kwargs):
    return ((img-img.mean())/img.std()).astype(np.float32)


def create_transform(image, label, size_smaller, size_larger, mean, std):

    mean = torch.tensor(mean).reshape(3, 1, 1)
    std = torch.tensor(std).reshape(3, 1, 1)
    
    #TODO: Add a condition to not resize if the size_smaller is None. 
    

    image = v2.functional.resize(
        image, 
        size=size_smaller, 
        max_size=size_larger,
        interpolation=v2.InterpolationMode.BILINEAR, 
        antialias=True
    )
    label = v2.functional.resize(
        label[None],
        size=size_smaller,
        max_size=size_larger,
        interpolation=v2.InterpolationMode.NEAREST_EXACT
    )[0]
    image = image.float()/255.
    image = (image-mean)/std

    return image, label



def create_datasets(img_dir, label_dir, crop_size=(256,256), train_val_split=0.1, seed=None):
    """Create dataset from directory. 
    
    Args
    crop_size: (height,width) to crop the images
    train_val_split: percentage of images used for validation
    use_simple: if True, use only crop, normalization and ToTensor. If False, use many data augmentation
    transformations.
    seed: seed for splitting the data
    """

    mean_data = 0.438
    std_data = 0.243

    # Set transformations
    train_transform = partial(create_transform, size_smaller=768, size_larger=2*768, mean=(mean_data, mean_data, mean_data), std=(std_data, std_data, std_data))
    valid_transform = partial(create_transform, size_smaller=768, size_larger=2*768, mean=(mean_data, mean_data, mean_data), std=(std_data, std_data, std_data))


    ds = ImageSegmentationDataset(img_dir, label_dir, name_to_label_map, img_opener=img_opener, label_opener=label_opener)
    ds_train, ds_valid = ds.split_train_val(train_val_split, seed=seed)
    ds_train.set_transform(train_transform)
    ds_valid.set_transform(valid_transform)

    return ds_train, ds_valid
