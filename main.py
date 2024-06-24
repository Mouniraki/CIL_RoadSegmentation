import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm

from augmentation import *
from mask_to_submission import *
from metrics import *
from models import *
from post_processing import *

PATCH_SIZE = 16  # pixels per side of square patches
VAL_SIZE = 10  # size of the validation set (number of images)
CUTOFF = 0.25  # minimum average brightness for a mask patch to be 

ROOT_PATH = "/home/xmarchon/CIL_RoadSegmentation"
test_path = "test"

"""
Data Augmentation
"""
run_augmentation_pipeline(5)

"""
images = load_all_from_path(os.path.join(ROOT_PATH, 'training', 'images'))[:, :, :, :3]
masks = load_all_from_path(os.path.join(ROOT_PATH, 'training', 'groundtruth'))

train_images, val_images, train_masks, val_masks = train_test_split(
    images, masks, test_size=0.2, random_state=42
)

show_first_n(train_images, train_masks)

# extract all patches and visualize those from the first image
train_patches, train_labels = image_to_patches(train_images, train_masks)
val_patches, val_labels = image_to_patches(val_images, val_masks)

# the first image is broken up in the first 25*25 patches
#show_patched_image(train_patches[:25*25], train_labels[:25*25])


test_path = os.path.join(ROOT_PATH, 'training', 'images')
test_filenames = sorted(glob(test_path + '/*.png'))
test_images = load_all_from_path(test_path)
test_patches = image_to_patches(test_images)


print("{0:0.2f}".format(sum(train_labels) / len(train_labels) * 100) + '% of training patches are labeled as 1.')
print("{0:0.2f}".format(sum(val_labels) / len(val_labels) * 100) + '% of validation patches are labeled as 1.')



device = 'cuda' if torch.cuda.is_available() else 'cpu'
# reshape the image to simplify the handling of skip connections and maxpooling
train_dataset = ImageDataset('training', device, use_patches=False, resize_to=(400, 400))
val_dataset = ImageDataset('validation', device, use_patches=False, resize_to=(400, 400))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)
model = UNet().to(device)
loss_fn = nn.BCELoss()
metric_fns = {'acc': accuracy_fn, 'patch_acc': patch_accuracy_fn}
optimizer = torch.optim.Adam(model.parameters())
n_epochs = 15 # default : 5
train(train_dataloader, val_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs)



# predict on test set
test_filenames = (glob(test_path + '/*.png'))
test_images = load_all_from_path(test_path)
batch_size = test_images.shape[0]
size = test_images.shape[1:3]
# we also need to resize the test images. This might not be the best ideas depending on their spatial resolution.
test_images = np.stack([cv2.resize(img, dsize=(384, 384)) for img in test_images], 0)
test_images = test_images[:, :, :, :3]
test_images = np_to_tensor(np.moveaxis(test_images, -1, 1), device)
test_pred = [model(t).detach().cpu().numpy() for t in test_images.unsqueeze(1)]
test_pred = np.concatenate(test_pred, 0)
test_pred= np.moveaxis(test_pred, 1, -1)  # CHW to HWC
test_pred = np.stack([cv2.resize(img, dsize=size) for img in test_pred], 0)  # resize to original shape
# now compute labels
test_pred = test_pred.reshape((-1, size[0] // PATCH_SIZE, PATCH_SIZE, size[0] // PATCH_SIZE, PATCH_SIZE))
test_pred = np.moveaxis(test_pred, 2, 3)
test_pred = np.round(np.mean(test_pred, (-1, -2)) > CUTOFF)
create_submission(test_pred, test_filenames, submission_filename='unet_submission.csv')




from collections import deque

def infer(img, kreis=3):
    point_set = {}
    model_out = model(img).detach().cpu().numpy()
    return model_out
    out_mask = model_out > 0.5 
    
    out_mask = mask_connected_though_border_radius(out_mask)
        
    return out_mask

images_train_for_metric = load_all_from_path(os.path.join(ROOT_PATH, 'training', 'images'))
masks_train_for_metric = load_all_from_path(os.path.join(ROOT_PATH, 'training', 'groundtruth'))

batch_size = images_train_for_metric.shape[0]
size = images_train_for_metric.shape[1:3]
# we also need to resize the test images. This might not be the best ideas depending on their spatial resolution.
images_train_for_metric = np.stack([cv2.resize(img, dsize=(400, 400)) for img in images_train_for_metric], 0)
images_train_for_metric = images_train_for_metric[:, :, :, :3]
images_train_for_metric = np_to_tensor(np.moveaxis(images_train_for_metric, -1, 1), device)

train_pred = [infer(t) for t in tqdm(images_train_for_metric.unsqueeze(1))]
train_pred = np.concatenate(train_pred, 0)[:,0,:,:]

images_train_for_metric= torch.permute(images_train_for_metric,(0,2,3,1))

train_patches_for_metric_1, train_labels_for_metric = image_to_patches(images_train_for_metric.detach().cpu().numpy(), train_pred)
train_patches_for_metric_2, train_true_label = image_to_patches(images_train_for_metric.cpu().numpy(), masks_train_for_metric)

print("Total F1 score without patch postprocessing   : %s" % (f1_score(train_labels_for_metric, train_true_label)))


for radius in range(0, 10):
    train_labels_for_metric_post_processing = patch_postprocessing(train_labels_for_metric, radius=radius)
    print("Total F1 score with patch postprocessing radius %s : %s" % (radius, f1_score(train_labels_for_metric_post_processing, train_true_label)))
    
    
for i in range(0,1):
    show_patched_image(train_patches_for_metric_1[i*25*25:(i+1)*25*25], train_labels_for_metric[i*25*25:(i+1)*25*25], figsize=(4, 4))
    show_patched_image(train_patches_for_metric_2[i*25*25:(i+1)*25*25], train_true_label[i*25*25:(i+1)*25*25], figsize=(4, 4))
    print("F1 score for frame %s : %s" % (i, f1_score(train_labels_for_metric[i*25*25:(i+1)*25*25], train_true_label[i*25*25:(i+1)*25*25])))
"""