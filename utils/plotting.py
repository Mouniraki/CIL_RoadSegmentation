import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np

def plot_patches(image, labels, d_patches=25):  
    fig, axs = plt.subplots(d_patches, d_patches, figsize=(18.5, 18.5))
    for i, (p, l) in enumerate(zip(image, labels)):
        # the np.maximum operation paints patches labeled as road red
        axs[i // d_patches, i % d_patches].imshow(torch.maximum(p/255, torch.tensor([l.item(), 0., 0.])))
        axs[i // d_patches, i % d_patches].set_axis_off()
    plt.show()

def show_val_samples(x, y, y_hat, segmentation=False):
    # training callback to show predictions on validation set
    imgs_to_draw = min(5, len(x))
    # if x.shape[-2:] == y.shape[-2:]:  # segmentation
    fig, axs = plt.subplots(3, imgs_to_draw, figsize=(18.5, 12))
    for i in range(imgs_to_draw):
        axs[0, i].imshow(x[i].moveaxis(0, -1))
        axs[1, i].imshow(torch.cat([y_hat[i].moveaxis(0, -1)] * 3, -1))
        axs[2, i].imshow(torch.cat([y[i].moveaxis(0, -1)]*3, -1))
        axs[0, i].set_title(f'Sample {i}')
        axs[1, i].set_title(f'Predicted {i}')
        axs[2, i].set_title(f'True {i}')
        axs[0, i].set_axis_off()
        axs[1, i].set_axis_off()
        axs[2, i].set_axis_off()
    # else:  # classification
    #     fig, axs = plt.subplots(1, imgs_to_draw, figsize=(18.5, 6))
    #     for i in range(imgs_to_draw):
    #         axs[i].imshow(x[i].moveaxis(0, -1))
    #         axs[i].set_title(f'True: {torch.round(y[i]).item()}; Predicted: {torch.round(y_hat[i]).item()}')
    #         axs[i].set_axis_off()
    plt.show()


# pytorch works with CHW format instead of HWC
def plot_img_mask(imgs, masks, nbr_of_pairs):
    fig, axes = plt.subplots(nbr_of_pairs, 2, figsize=(10, 5 * nbr_of_pairs))
    
    for i in range(nbr_of_pairs):
        img = imgs[i]
        mask = masks[i]
      
        axes[i, 0].imshow(img.moveaxis(0, -1))
        axes[i, 0].set_title(f'Image {i+1}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask.moveaxis(0, -1),  cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Mask {i+1}')
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    plt.show()