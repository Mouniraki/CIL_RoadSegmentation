import os
from datetime import datetime # To save the predicted masks in a dated folder
from tqdm import tqdm

# PyTorch imports
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torchvision.io import write_png
from torchvision.transforms.functional import adjust_contrast

# Importing the dataset & models
from utils.loaders.image_dataset import ImageDataset
from utils.loaders.transforms import compose, colorjitter
from utils.models.unet import UNet
from utils.models.segformer import SegFormer

# Importing plot & metric utilities
from utils.plotting import plot_patches, show_val_samples
from utils.metrics import patch_accuracy_fn, iou_fn, precision_fn, recall_fn, f1_fn

# To select the proper hardware accelerator
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Constants
PATCH_SIZE = 16
CUTOFF = 0.25

TRAIN_SPLIT = 0.8
BATCH_SIZE = 4
N_WORKERS = 4 # Base is 4, set to 0 if it causes errors
N_EPOCHS = 5

def main():
    print(f"Using {DEVICE} device")

    # Initializing Tensorboard
    writer = SummaryWriter(log_dir='tensorboard/')

    # Setting up the model, loss function and optimizer
    # model = UNet().to(DEVICE)
    # loss_fn = torch.nn.BCELoss()
    model = SegFormer(non_void_labels=['road'], checkpoint='nvidia/mit-b5').to(DEVICE)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)

    #############################
    # Training routine
    #############################
    print("Starting training")
    # Selecting the transformations to perform for data augmentation
    transforms = compose.Compose([
        colorjitter.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
    ])
    # Loading the whole training dataset
    images_dataset = ImageDataset(
        for_train=True,
        images_dir = os.path.abspath('dataset/training/images/'),
        masks_dir = os.path.abspath('dataset/training/groundtruth/'),
        transforms=transforms,
        use_patches=False,
        # img_size=(512, 512)
    )

    # Performing a train/validation split
    train_dataset, val_dataset = random_split(images_dataset, [TRAIN_SPLIT, 1 - TRAIN_SPLIT])

    # Creating a DataLoader for each of the train and validation datasets, used to load batches of images from the respective datasets
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=True)
    validation_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=True)

    for epoch in range(N_EPOCHS):
        # For the progress bar (and to load the images from the mini-batch)
        progress_bar = tqdm(iterable=train_dataloader, desc=f"Epoch {epoch+1} / {N_EPOCHS}")
        # Perform training
        model.train()
        for (x, y) in progress_bar: # x = images, y = labels
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad() # Zero-out gradients
            y_hat = model(x) # Forward pass
            # y_hat = adjust_contrast(y_hat, contrast_factor=0.5) # Force the predictions to be more contrasted
            loss = loss_fn(y_hat, y)
            if type(loss_fn) == torch.nn.modules.loss.BCEWithLogitsLoss:
                y_hat = torch.sigmoid(y_hat)
            writer.add_scalar("Loss/train", loss.item(), epoch)
            loss.backward() # Backward pass
            optimizer.step()
        
        # Perform validation
        model.eval()
        with torch.no_grad():
            # To compute the overall patch accuracy
            batch_patch_acc, batch_iou, batch_precision, batch_recall, batch_f1 = [], [], [], [], []
            for (x, y) in validation_dataloader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                y_hat = model(x) # Perform forward pass
                loss = loss_fn(y_hat, y)
                # Apply a sigmoid to the inference result if we use the BCEWithLogitsLoss for metrics computations
                if type(loss_fn) == torch.nn.modules.loss.BCEWithLogitsLoss:
                    y_hat = torch.sigmoid(y_hat)
                writer.add_scalar("Loss/eval", loss.item(), epoch)

                batch_patch_acc.append(patch_accuracy_fn(y_hat=y_hat, y=y))
                batch_iou.append(iou_fn(y_hat=y_hat, y=y))
                batch_precision.append(precision_fn(y_hat=y_hat, y=y))
                batch_recall.append(recall_fn(y_hat=y_hat, y=y))
                batch_f1.append(f1_fn(y_hat=y_hat, y=y))
            
            # Computing the metrics
            patch_acc = torch.cat(batch_patch_acc, dim=0).mean()
            mean_iou = torch.cat(batch_iou, dim=0).mean()
            precision = torch.cat(batch_precision, dim=0).mean()
            recall = torch.cat(batch_recall, dim=0).mean()
            f1 = torch.cat(batch_f1, dim=0).mean()

            writer.add_scalar(f"Accuracy/eval", patch_acc, epoch)
            writer.add_scalar(f"Mean IoU/eval", mean_iou, epoch)
            writer.add_scalar(f"Precision/eval", precision, epoch)
            writer.add_scalar(f"Recall/eval", recall, epoch)
            writer.add_scalar(f"F1 score/eval", f1, epoch)

            print(f"Overall patch accuracy: {patch_acc}")
            print(f"Mean IoU: {mean_iou}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
        
        # Optional : display the validation samples used for validation
        show_val_samples(x.detach().cpu(), y.detach().cpu(), y_hat.detach().cpu())


    #############################
    # Inference routine
    #############################
    print("Performing inference")
    test_dataset = ImageDataset(
        for_train=False,
        images_dir = os.path.abspath('dataset/test/images/'),
        # img_size = (512, 512)
    )
    # We don't shuffle to keep the original data ordering
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=False)

    # Create a new folder for the predicted masks
    curr_date = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    os.mkdir(f'predictions/{curr_date}')

    model.eval()
    with torch.no_grad():
        img_idx = 144 # Test images start at index 144
        for x, _ in test_dataloader:
            x = x.to(DEVICE)
            pred = model(x).detach().cpu()
            if type(loss_fn) == torch.nn.modules.loss.BCEWithLogitsLoss:
                pred = torch.sigmoid(pred)
            # Add channels to end up with RGB tensors, and save the predicted masks on disk
            pred = torch.cat([pred.moveaxis(1, -1)]*3, -1).moveaxis(-1, 1) # Here the dimension 0 is for the number of images, since we feed a batch!
            for t in pred:
                t = (t * 255).type(torch.uint8) # Rescale everything in the 0-255 range to generate channels for images
                write_png(input=t, filename=f'predictions/{curr_date}/satimage_{img_idx}.png')
                img_idx += 1

    writer.flush()
    writer.close()

main()