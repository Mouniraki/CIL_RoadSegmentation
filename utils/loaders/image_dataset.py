import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class ImageDataset(Dataset):
    def __init__(self, for_train: bool, images_dir: str, masks_dir: str | None = None, use_patches: bool = False, patch_size: int=16, cutoff: float=0.25):
        self.for_train = for_train
        self.images_dir = images_dir
        self.masks_dir = masks_dir
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
        image = read_image(img_path)[:3,:,:] / 255 # (channels, height, width) with values as floating points between 0 and 1

        # Handling the case of the mask
        if self.masks_dir != None:
            mask_path = os.path.join(self.masks_dir, filename)
            mask = read_image(mask_path) / 255
        else:
            mask = None
        
        if self.use_patches:
            image = self.__split_in_patches(img=image).reshape(3, -1, self.patch_size, self.patch_size)
            labels = None if mask == None else (torch.mean(self.__split_in_patches(img=mask), (0, -1, -2), dtype=torch.float32) > self.cutoff).reshape(-1)
        else:
            labels = mask

        return image, labels if labels != None else image
        # else:
        #     if self.masks_dir != None:
        #         return 

        # if self.masks_dir != None:
        #     mask_path = os.path.join(self.masks_dir, filename)
        #     mask = read_image(mask_path)
        #     if self.use_patches:
        #         mask = self.__split_in_patches(img=mask)
        #         labels = torch.mean(mask, (0, -1, -2), dtype=torch.float32) > self.cutoff
        #         labels = labels.reshape(-1)
        #     else:
        #         labels = mask / 255 # Normalize everything between 0 and 1
        # else:

        

        # if self.use_patches:
        #     img_patches = self.__split_in_patches(img=image).reshape(3, -1, self.patch_size, self.patch_size)

        #     # Loading a mask, splitting it into patches and creating labels for each image patch
        #     if self.masks_dir != None:
        #         mask_path = os.path.join(self.masks_dir, filename)
        #         mask = read_image(mask_path).reshape((-1, h_patches, self.patch_size, w_patches, self.patch_size))
        #         mask = mask.moveaxis(2, 3)
        #         labels = torch.mean(mask, (0, -1, -2), dtype=torch.float32) > self.cutoff
        #         labels = labels.reshape(-1)
        #         return img_patches, labels
        #     else:
        #         return img_patches
        # else:
        #     if self.masks_dir != None:
        #         mask_path = os.path.join(self.masks_dir, filename)
        #         labels = read_image(mask_path) / 255
        #         return image, labels
        #     else:
        #         return image
    
    def __split_in_patches(self, img: torch.Tensor):
        _, h, w = img.shape
        h_patches = h // self.patch_size
        w_patches = w // self.patch_size
        patches = img.reshape((-1, h_patches, self.patch_size, w_patches, self.patch_size))
        patches = patches.moveaxis(2, 3)
        return patches