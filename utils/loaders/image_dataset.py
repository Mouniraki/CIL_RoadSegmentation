import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.v2 import Resize

class ImageDataset(Dataset):
    def __init__(self, 
                 for_train: bool, 
                 images_dir: str, 
                 masks_dir: str | None = None, 
                 img_size: tuple[int, int] = (400, 400), # Default size is (400, 400), otherwise we resize
                 use_patches: bool = False, 
                 patch_size: int=16, 
                 cutoff: float=0.25):
        self.for_train = for_train
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.desired_img_h, self.desired_img_w = img_size
        self.use_patches = use_patches
        self.patch_size = patch_size
        self.cutoff = cutoff
        self.len = len([f for f in os.listdir(self.images_dir) if os.path.isfile(self.images_dir + '/' + f)])

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if not self.for_train:
            idx = idx + 144 # Indices of testing images start from 144 instead of 0
        filename = f"satimage_{idx}.png"
        img_path = os.path.join(self.images_dir, filename)

        # Loading an image & splitting it into patches (IMAGES HAVE 4 CHANNELS ORIGINALLY (r,g,b, alpha), PICK ONLY THE FIRST 3!)
        image = read_image(img_path)[:3,:,:] # (channels, height, width) with values as floating points between 0 and 1
        _, h, w = image.shape
        if self.desired_img_h != h or self.desired_img_w != w:
            # Resize the image
            image = Resize(size=(self.desired_img_h, self.desired_img_w)).forward(image)
        image = image / 255

        # Handling the case of the mask
        if self.masks_dir != None:
            mask_path = os.path.join(self.masks_dir, filename)
            mask = read_image(mask_path)
            if self.desired_img_h != h or self.desired_img_w != w:
                mask = Resize(size=(self.desired_img_h, self.desired_img_w)).forward(mask)
            mask = mask / 255
        else:
            mask = None
        
        if self.use_patches:
            image = self.__split_in_patches(img=image).reshape(3, -1, self.patch_size, self.patch_size)
            labels = None if mask == None else (torch.mean(self.__split_in_patches(img=mask), (0, -1, -2), dtype=torch.float32) > self.cutoff).reshape(-1)
        else:
            labels = mask

        return image, labels if labels != None else image
    
    def __split_in_patches(self, img: torch.Tensor):
        _, h, w = img.shape
        h_patches = h // self.patch_size
        w_patches = w // self.patch_size
        patches = img.reshape((-1, h_patches, self.patch_size, w_patches, self.patch_size))
        patches = patches.moveaxis(2, 3)
        return patches