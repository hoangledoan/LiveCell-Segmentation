import os
import random
from PIL import Image
import numpy as np
import torch
import cv2
# pylint: disable=E1101


def elastic_transform(image, alpha, sigma, random_state=None):
    """Function using for elastic tranformation the dataset."""
    if random_state is None:
        random_state = np.random.RandomState()

    shape_size = image.shape[:2]
    # Downscaling the random grid and then upsizing post filter
    # improves performance. Approx 3x for scale of 4, diminishing 
    # returns after.
    grid_scale = 4
    alpha //= grid_scale
    sigma //= grid_scale
    grid_shape = (shape_size[0]//grid_scale, shape_size[1]//grid_scale)

    blur_size = int(4 * sigma) | 1
    rand_x = cv2.GaussianBlur(
        (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
        ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
    rand_y = cv2.GaussianBlur(
        (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
        ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
    if grid_scale > 1:
        rand_x = cv2.resize(rand_x, shape_size[::-1])
        rand_y = cv2.resize(rand_y, shape_size[::-1])

    grid_x, grid_y = np.meshgrid(np.arange(shape_size[1]), 
                                 np.arange(shape_size[0]))
    grid_x = (grid_x + rand_x).astype(np.float32)
    grid_y = (grid_y + rand_y).astype(np.float32)

    distorted_img = cv2.remap(image, grid_x, grid_y,
                              borderMode=cv2.BORDER_REFLECT_101, 
                              interpolation=cv2.INTER_LINEAR)

    return distorted_img


class Dataset():
    """Load and transform the dataset."""
    def __init__(self, image_dir, mask_dir, transform=None, train=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.train = train
        self.images = os.listdir(image_dir)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.train is True:
            img_path = os.path.join(self.image_dir, self.images[index])
            mask_path = os.path.join(self.mask_dir, self.images[index])

            angle = random.randint(0, 180)
            
            image = Image.open(img_path)
            # Rotate the image
            image = image.rotate(angle)
            # Convert the image to RGB image and crop
            image = np.array(image.convert('RGB'))
            image = image[:333, :434, :]                
            
            mask = Image.open(mask_path)
            # Rotate the mask with the same angle
            mask = mask.rotate(angle)
            # Convert the mask to binary image and crop
            mask = np.array(mask.convert('L'))
            mask = mask[:333, :434]
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask) 
                
                image = torch.tensor(image)
                mask = torch.tensor(mask)
                
                image = image.permute(1, 2, 0).numpy()
                mask = mask.squeeze(0).numpy()
                    
                # Apply elastic transformation with the fix seed for 
                # both image and mask
                random_state = np.random.RandomState(50)
                image = elastic_transform(image, mask.shape[1] * 2, mask.shape[1] * 0.08, random_state)
                mask = elastic_transform(mask, mask.shape[1] * 2, mask.shape[1] * 0.08, random_state)
                
                image = torch.tensor(image).permute(2, 0, 1)
                mask = torch.tensor(mask).unsqueeze(0)
                    
        # Normal transform for validation and test dataset        
        else:
            img_path = os.path.join(self.image_dir, self.images[index])
            mask_path = os.path.join(self.mask_dir, self.images[index])
            
            # Convert the image to RGB image and crop
            image = Image.open(img_path)
            image = np.array(image.convert('RGB'))
            image = image[:333, :434, :]  
            
            # Convert the mask to binary image and crop
            mask = Image.open(mask_path)
            mask = np.array(mask.convert('L'))
            mask = mask[:333, :434]
            
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
        
        return image, mask