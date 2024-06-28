import os
import cv2
import numpy as np
from sklearn.metrics import f1_score
import torch
from torch import nn
from tqdm import tqdm

from dataset import Data
from metrics import accuracy_fn, patch_accuracy_fn
from models import UNet
from post_processing import patch_postprocessing, algos_and_params
from utils import show_first_n, image_to_patches, train, create_submission, np_to_tensor, \
    show_patched_image

PATCH_SIZE = 16  # pixels per side of square patches
VAL_SIZE = 10  # size of the validation set (number of images)
CUTOFF = 0.25  # minimum average brightness for a mask patch to be 

ROOT_PATH = "/home/xmarchon/CIL_RoadSegmentation"
ROOT_PATH = "./"
path_to_train_data = os.path.join(ROOT_PATH, 'training')
path_to_test_data = os.path.join(ROOT_PATH, 'test', 'images')

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    print("No GPU available, using CPU instead.")

"""
load the images and Data Augmentation
"""
data = Data(path_to_train_data, path_to_test_data, device, with_augmentation_nbr=0, patch_size=PATCH_SIZE,
            cutoff=CUTOFF)

show_first_n(data.train_images, data.train_masks)
# extract all patches and visualize those from the first image
print("{0:0.2f}".format(
    sum(data.train_labels) / len(data.train_labels) * 100) + '% of training patches are labeled as 1.')
print(
    "{0:0.2f}".format(sum(data.val_labels) / len(data.val_labels) * 100) + '% of validation patches are labeled as 1.')

# reshape the image to simplify the handling of skip connections and maxpooling

model = UNet().to(device)
loss_fn = nn.BCELoss()
metric_fns = {'acc': accuracy_fn, 'patch_acc': patch_accuracy_fn}
optimizer = torch.optim.Adam(model.parameters())
n_epochs = 5  # default : 5
train(data.train_dataloader, data.val_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs)

# predict on test set
batch_size = data.test_images.shape[0]
size = data.test_images.shape[1:3]
# we also need to resize the test images. This might not be the best ideas depending on their spatial resolution.
test_images = np.stack([cv2.resize(img, dsize=(384, 384)) for img in data.test_images], 0)
test_images = test_images[:, :, :, :3]
test_images = np_to_tensor(np.moveaxis(test_images, -1, 1), device)
test_pred = [model(t).detach().cpu().numpy() for t in test_images.unsqueeze(1)]
test_pred = np.concatenate(test_pred, 0)
test_pred = np.moveaxis(test_pred, 1, -1)  # CHW to HWC
test_pred = np.stack([cv2.resize(img, dsize=size) for img in test_pred], 0)  # resize to original shape
# now compute labels
test_pred = test_pred.reshape((-1, size[0] // PATCH_SIZE, PATCH_SIZE, size[0] // PATCH_SIZE, PATCH_SIZE))
test_pred = np.moveaxis(test_pred, 2, 3)
test_pred = np.round(np.mean(test_pred, (-1, -2)) > CUTOFF)
create_submission(test_pred, path_to_test_data, data.test_filenames, submission_filename='unet_submission.csv')




def infer(img, kreis=3):
    point_set = {}
    model_out = model(img).detach().cpu().numpy()
    return model_out
    out_mask = model_out > 0.5
    out_mask = mask_connected_though_border_radius(out_mask)
    return out_mask


batch_size = data.train_images.shape[0]
size = data.train_masks.shape[1:3]
# we also need to resize the test images. This might not be the best ideas depending on their spatial resolution.
images_train_for_metric = np.stack([cv2.resize(img, dsize=(400, 400)) for img in data.train_images], 0)
images_train_for_metric = images_train_for_metric[:, :, :, :3]
images_train_for_metric = np_to_tensor(np.moveaxis(images_train_for_metric, -1, 1), device)

train_pred = [infer(t) for t in tqdm(images_train_for_metric.unsqueeze(1))]
train_pred = np.concatenate(train_pred, 0)[:, 0, :, :]

images_train_for_metric = torch.permute(images_train_for_metric, (0, 2, 3, 1))

train_patches_for_metric_1, train_labels_for_metric = image_to_patches(images_train_for_metric.detach().cpu().numpy(),
                                                                       patch_size=PATCH_SIZE, cutoff=CUTOFF,
                                                                       masks=train_pred)
train_patches_for_metric_2, train_true_label = image_to_patches(images_train_for_metric.cpu().numpy(),
                                                                patch_size=PATCH_SIZE, cutoff=CUTOFF,
                                                                masks=data.train_masks)

print("Total F1 score without patch postprocessing   : %s" % (f1_score(train_labels_for_metric, train_true_label)))


exit()
#for radius in range(0, 10):
#    train_labels_for_metric_post_processing = patch_postprocessing(train_labels_for_metric, algorithm='mask_connected_though_border_radius', radius=radius)
#    print("Total F1 score with patch postprocessing radius %s : %s" % (
#    radius, f1_score(train_labels_for_metric_post_processing, train_true_label)))



# Test postprocessing
#mask_connected_though_border_radius
for radius in range(0, 15, 3):
        algo_and_params = algos_and_params['mask_connected_though_border_radius']
        algo_and_params['radius'] = radius
        train_labels_for_metric_post_processing = patch_postprocessing(train_labels_for_metric, algo_and_params)
        print(f"Total F1 score with patch postprocessing "
              f"mask_connected_though_border_radius(radius={radius}) : {f1_score(train_labels_for_metric_post_processing, train_true_label)}")

#extend_path_to_closest
#algo_and_params = algos_and_params['extend_path_to_closest']
#train_labels_for_metric_post_processing = patch_postprocessing(train_labels_for_metric, algo_and_params)
#print(f"Total F1 score with patch postprocessing "
#      f"extend_path_to_closest : {f1_score(train_labels_for_metric_post_processing, train_true_label)}")
#connect_road
for min_group_size in range(0, 12, 4):
    for max_dist in range(0,25, 5):
        algo_and_params = algos_and_params['connect_road']
        algo_and_params['min_group_size'], algo_and_params['max_dist'] = min_group_size, max_dist
        train_labels_for_metric_post_processing = patch_postprocessing(train_labels_for_metric, algo_and_params)
        print(f"Total F1 score with patch postprocessing "
              f"connect_road(min_group_size={min_group_size}, "
              f"max_dist={max_dist}) : {f1_score(train_labels_for_metric_post_processing, train_true_label)}")






for i in range(0, 3):
    show_patched_image(train_patches_for_metric_1[i * 25 * 25:(i + 1) * 25 * 25],
                       train_labels_for_metric_post_processing[i * 25 * 25:(i + 1) * 25 * 25])
    show_patched_image(train_patches_for_metric_1[i * 25 * 25:(i + 1) * 25 * 25],
                       train_labels_for_metric[i * 25 * 25:(i + 1) * 25 * 25])
    show_patched_image(train_patches_for_metric_2[i * 25 * 25:(i + 1) * 25 * 25],
                       train_true_label[i * 25 * 25:(i + 1) * 25 * 25])
    print("F1 score for frame %s : %s" % (i, f1_score(train_labels_for_metric[i * 25 * 25:(i + 1) * 25 * 25],
                                                      train_true_label[i * 25 * 25:(i + 1) * 25 * 25])))
