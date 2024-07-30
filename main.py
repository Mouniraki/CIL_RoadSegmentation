import argparse
import os
from datetime import datetime # To save the predicted masks in a dated folder
from tqdm import tqdm

# PyTorch imports
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split, RandomSampler
from torchvision.io import write_png

# Importing the dataset & models
from utils.loaders.image_dataset import ImageDataset
from utils.loaders.transforms import compose, rotation, colorjitter, randomerasing, randomresizedcrop
from utils.models.unet import UNet
from utils.models.segformer import SegFormer
from utils.losses.loss import DiceLoss, BorderDiceLoss, CombineLoss

# Importing plot & metric utilities
from utils.plotting import plot_patches, show_val_samples
from utils.metrics import patch_accuracy_fn, iou_fn, precision_fn, recall_fn, f1_fn, patch_f1_fn

# To select the proper hardware accelerator
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Constants
PATCH_SIZE = 16
CUTOFF = 0.25

parser = argparse.ArgumentParser(prog='main', description='The file implement the trainig loop for our CIL project implementation')
parser.add_argument('-ne', '--n_epochs', help='maximum number of epochs performed during training', default=100, type=int)
parser.add_argument('-na', '--n_augmentation', help='The number of pass on the dataset with different transformations perfomed at each training iteration', default=4, type=int) # Set to 1 for only 1 pass
parser.add_argument('-s', '--early_stopping_threshold', help='Nbr of epoch given to the model to improve on previously better result', default=10, type=int)
parser.add_argument('-bs', '--batch_size', help='The nbr of sample evaluated in parallel ', default=4, type=int)
parser.add_argument('-d', '--debug', help=' To enable / disable the show_val_samples routine ', default=True)
parser.add_argument('-m', '--model', help='Set this to the desired model', default="segformer")
args = parser.parse_args()


TRAIN_SPLIT = 0.8
LR = 0.00006
N_WORKERS = 4 # Base is 4, set to 0 if it causes errors

# To create folders for test predictions and model checkpoints
CURR_DATE = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
CHECKPOINTS_FOLDER = f"checkpoints/{CURR_DATE}"
TENSORBOARD_FOLDER = f"tensorboard/{CURR_DATE}"
INFERENCE_FOLDER = f"predictions/{CURR_DATE}"

TRAIN_DATASET_PATH = 'dataset/training'
TEST_DATASET_PATH = 'dataset/test/images/'
CHECKPOINTS_FILE_PREFIX = 'epoch'
INFERENCE_FILE_PREFIX = 'satimage'


def main():
    print(f"Using {DEVICE} device")

    # Initializing Tensorboard
    writer = SummaryWriter(log_dir=TENSORBOARD_FOLDER)

    # Setting up the model, loss function and optimizer
    if args.model == 'segformer':
        model = SegFormer(non_void_labels=['road'], checkpoint='nvidia/mit-b5').to(DEVICE)
        # loss_fn = torch.nn.BCEWithLogitsLoss()
        loss_fn = DiceLoss(model=args.model)
    else:
        model = UNet().to(DEVICE)
        loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    #############################
    # Training routine
    #############################
    # Create a directory for model checkpoints
    os.makedirs(CHECKPOINTS_FOLDER, exist_ok=True)

    print("Starting training")
    # Selecting the transformations to perform for data augmentation
    transforms = compose.Compose([rotation.Rotation(angle=30, probability=0.6),
                                    colorjitter.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                    randomerasing.RandomErasing(probability=0.5, scale=(0.01, 0.2), ratio=(0.3, 3.3))])
    
    # Loading the whole training dataset
    images_dataset = ImageDataset(
        for_train=True,
        images_dir = os.path.abspath(f'{TRAIN_DATASET_PATH}/images/'),
        masks_dir = os.path.abspath(f'{TRAIN_DATASET_PATH}/groundtruth/'),
        transforms=transforms,
        use_patches=False,
        # img_size=(512, 512)
    )

    # Performing a training/validation split
    train_dataset, val_dataset = random_split(images_dataset, [TRAIN_SPLIT, 1 - TRAIN_SPLIT])

    # Creating a DataLoader for each of the training and validation datasets, used to load batches of images from the respective datasets
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=N_WORKERS, shuffle=True)
    validation_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=N_WORKERS, shuffle=True)

    # Early stopping mechanism
    best_epoch = 0
    best_loss = 100

    for epoch in range(args.n_epochs):
        # Perform training
        model.train()
        losses = [] # To record metric
        # Perform data augmentation by re-feeding n times the training dataset with random transformations each time
        for n_a in range(args.n_augmentation):
            #random_sampler = RandomSampler(train_dataloader.dataset, replacement=False, num_samples=400) # can be used to reduce load while using a big dataset for testing
            #sampler_dataloader = DataLoader(train_dataloader.dataset, sampler=random_sampler, batch_size=train_dataloader.batch_size)
            # For the progress bar (and to load the images from the mini-batch)
            progress_bar = tqdm(iterable=train_dataloader, desc=f"Epoch {epoch+1} / {args.n_epochs} <- Augmentation : {n_a+1} / {args.n_augmentation}")
            for (x, y) in progress_bar: # x = images, y = labels
                x = x.to(DEVICE)
                y = y.to(DEVICE).mean(dim=1).unsqueeze(1)
                optimizer.zero_grad() # Zero-out gradients
                y_hat = model(x) # Forward pass
                # y_hat = adjust_contrast(y_hat, contrast_factor=0.5) # Force the predictions to be more contrasted
                loss = loss_fn(y_hat, y)
                losses.append(loss.item())
                if args.model == 'segformer':
                    y_hat = torch.sigmoid(y_hat)
                loss.backward() # Backward pass
                optimizer.step()

        mean_loss = torch.tensor(losses).mean()
        writer.add_scalar("Loss/training", mean_loss.item(), epoch)

        # Perform validation
        model.eval()
        with torch.no_grad():
            # To compute the overall patch accuracy
            batch_patch_acc, batch_iou, batch_precision, batch_recall, batch_f1, batch_patch_f1 = [], [], [], [], [], []

            losses = [] # For the early stopping mechanism
            progress_bar_validation_dataloader = tqdm(iterable=validation_dataloader, desc=f"Epoch {epoch+1} / {args.n_epochs} <- Augmentation : {n_a+1} / {args.n_augmentation}")
            for (x, y) in progress_bar_validation_dataloader:
                x = x.to(DEVICE)
                y = y.to(DEVICE).mean(dim=1).unsqueeze(1)
                y_hat = model(x) # Perform forward pass
                loss = loss_fn(y_hat, y)
                # Apply a sigmoid to the inference result if we choose SegFormer as a model
                if args.model == 'segformer':
                    y_hat = torch.sigmoid(y_hat)

                losses.append(loss.item())
                batch_patch_acc.append(patch_accuracy_fn(y_hat=y_hat, y=y))
                batch_iou.append(iou_fn(y_hat=y_hat, y=y))
                batch_precision.append(precision_fn(y_hat=y_hat, y=y))
                batch_recall.append(recall_fn(y_hat=y_hat, y=y))
                batch_f1.append(f1_fn(y_hat=y_hat, y=y))
                batch_patch_f1.append(patch_f1_fn(y_hat=y_hat, y=y))

            # Computing the metrics
            mean_loss = torch.tensor(losses).mean()
            patch_acc = torch.cat(batch_patch_acc, dim=0).mean()
            mean_iou = torch.cat(batch_iou, dim=0).mean()
            precision = torch.cat(batch_precision, dim=0).mean()
            recall = torch.cat(batch_recall, dim=0).mean()
            f1 = torch.cat(batch_f1, dim=0).mean()
            patch_f1 = torch.cat(batch_patch_f1, dim=0).mean()

            writer.add_scalar("Loss/eval", mean_loss.item(), epoch)
            writer.add_scalar(f"Accuracy/eval", patch_acc, epoch)
            writer.add_scalar(f"Mean IoU/eval", mean_iou, epoch)
            writer.add_scalar(f"Precision/eval", precision, epoch)
            writer.add_scalar(f"Recall/eval", recall, epoch)
            writer.add_scalar(f"F1 score/eval", f1, epoch)
            writer.add_scalar(f"Patch F1 score/eval", patch_f1, epoch)

            print(f"Overall patch accuracy: {patch_acc}")
            print(f"Mean IoU: {mean_iou}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            print(f"Patch F1 Score: {patch_f1}")
            print(f"Loss: {mean_loss}")

            # Optional : display the validation samples used for validation
            if args.debug:
                show_val_samples(x.detach().cpu(), y.detach().cpu(), y_hat.detach().cpu())

            if mean_loss <= best_loss:
                best_loss = mean_loss
                best_epoch = epoch
                # Save a checkpoint if the best epoch is greater than 10 (avoid saving too much checkpoints)
                # if epoch >= 5:
                model.save_pretrained(f"{CHECKPOINTS_FOLDER}/{CHECKPOINTS_FILE_PREFIX}-{best_epoch+1}.pth")
                # torch.save(model.state_dict, f"checkpoints/{CURR_DATE}/epoch-{best_epoch+1}.pth")
            elif epoch - best_epoch >= args.early_stopping_threshold:
                print(f"Early stopped at epoch {epoch+1} with best epoch {best_epoch+1}")
                break
    
    writer.flush()
    writer.close()

    #############################
    # Inference routine
    #############################
    # Load best model from checkpoint
    model = model.load_pretrained(checkpoint=f"{CHECKPOINTS_FOLDER}/{CHECKPOINTS_FILE_PREFIX}-{best_epoch+1}.pth").to(DEVICE)

    print("Performing inference")
    test_dataset = ImageDataset(
        for_train=False,
        images_dir = os.path.abspath(TEST_DATASET_PATH),
        # img_size = (512, 512)
    )
    # We don't shuffle to keep the original data ordering
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=N_WORKERS, shuffle=False)

    # Create a new folder for the predicted masks
    os.makedirs(INFERENCE_FOLDER, exist_ok=True)

    model.eval()
    with torch.no_grad():
        img_idx = 144 # Test images start at index 144
        for x, _ in test_dataloader:
            x = x.to(DEVICE)
            pred = model(x).detach().cpu()
            if args.model == 'segformer':
                pred = torch.sigmoid(pred)
            # Add channels to end up with RGB tensors, and save the predicted masks on disk
            pred = torch.cat([pred.moveaxis(1, -1)]*3, -1).moveaxis(-1, 1) # Here the dimension 0 is for the number of images, since we feed a batch!
            for t in pred:
                t = (t * 255).type(torch.uint8) # Rescale everything in the 0-255 range to generate channels for images
                write_png(input=t, filename=f'{INFERENCE_FOLDER}/{INFERENCE_FILE_PREFIX}_{img_idx}.png')
                img_idx += 1


if __name__ == "__main__":
    main()
