import os

import torch

from augmentation import run_augmentation_pipeline
from utils import np_to_tensor, load_all_from_path, image_to_patches
from glob import glob
from sklearn.model_selection import train_test_split
import numpy as np
import cv2


class Data():
    def __init__(self, path_to_train_data, path_to_test_data, device, with_augmentation_nbr=0, patch_size=16, cutoff=0.25):
        self.device = device
        #if with_augmentation_nbr > 0:
        #    run_augmentation_pipeline(with_augmentation_nbr)
        #    self.train_images = load_all_from_path(os.path.join(path_to_train_data, 'images', 'output'))[:, :, :, :3]
        #else:
        self.train_images = load_all_from_path(os.path.join(path_to_train_data, 'images'))[:, :, :, :3]
        self.train_masks = load_all_from_path(os.path.join(path_to_train_data, 'groundtruth'))
        self.train_images, self.val_images, self.train_masks, self.val_masks = train_test_split(
            self.train_images, self.train_masks, test_size=0.2, random_state=42
        )
        self.train_patches, self.train_labels = image_to_patches(self.train_images, patch_size, cutoff, masks=self.train_masks)
        self.val_patches, self.val_labels = image_to_patches(self.val_images, patch_size, cutoff, masks=self.val_masks)
        self.test_filenames = sorted(glob(path_to_test_data + '/*.png'))
        self.test_images = load_all_from_path(path_to_test_data)
        self.test_patches = image_to_patches(self.test_images, patch_size, cutoff)

        self.train_dataset = ImageDataset(self.train_images, self.train_masks, device, use_patches=False, resize_to=(400, 400), patch_size=patch_size, cutoff=cutoff)
        self.val_dataset = ImageDataset( self.val_images, self.val_masks , device, use_patches=False, resize_to=(400, 400), patch_size=patch_size, cutoff=cutoff)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=4, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=4, shuffle=True)






class ImageDataset(torch.utils.data.Dataset):
    # dataset class that deals with loading the data and making it available by index.

    def __init__(self, images, masks, device, use_patches=True, resize_to=(400, 400), patch_size=16, cutoff=0.25):
        self.patch_size = patch_size
        self.cutoff = cutoff
        self.images = images
        self.masks = masks
        self.device = device
        self.use_patches = use_patches
        self.resize_to=resize_to
        self.x, self.y, self.n_samples = None, None, None
        self._load_data()

    def _load_data(self):  # not very scalable, but good enough for now
        self.x = self.images
        self.y = self.masks
        if self.use_patches:  # split each image into patches
            self.x, self.y = image_to_patches(self.x, self.patch_size, self.cutoff, masks=self.y)
        elif self.resize_to != (self.x.shape[1], self.x.shape[2]):  # resize images
            self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x], 0)
            self.y = np.stack([cv2.resize(mask, dsize=self.resize_to) for mask in self.y], 0)
        self.x = np.moveaxis(self.x, -1, 1)  # pytorch works with CHW format instead of HWC
        self.n_samples = len(self.x)

    def _preprocess(self, x, y):
        # to keep things simple we will not apply transformations to each sample,
        # but it would be a very good idea to look into preprocessing
        return x, y

    def __getitem__(self, item):
        return self._preprocess(np_to_tensor(self.x[item], self.device), np_to_tensor(self.y[[item]], self.device))

    def __len__(self):
        return self.n_samples