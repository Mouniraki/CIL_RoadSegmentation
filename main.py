import argparse
import os
from datetime import datetime # To save the predicted masks in a dated folder
from tqdm import tqdm

# PyTorch imports
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.io import write_png

# For the K-fold cross-validation & single train/test splits
from sklearn.model_selection import KFold

# Importing the dataset & models
from utils.loaders.image_dataset import ImageDataset
from utils.loaders.transforms import compose, rotation, colorjitter, randomerasing
from utils.models.segformer import SegFormer
from utils.losses.loss import DiceLoss

# Importing plot & metric utilities
from utils.plotting import show_val_samples
from utils.metrics import patch_accuracy_fn, iou_fn, precision_fn, recall_fn, f1_fn, patch_f1_fn

# To select the proper hardware accelerator
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Constants
PATCH_SIZE = 16
CUTOFF = 0.25

parser = argparse.ArgumentParser(prog='main', description='The file implement the training loop of the main model for our CIL project implementation')
parser.add_argument('--n_folds', choices=range(3, 6), help='options: [3 (for a 67/33 split), 4 (for a 75/25 split), 5 (for a 80/20 split)]', type=int, default=5)
parser.add_argument('--single_split_train', help='Train on a single train/validation split of the K-fold splits', action='store_true')
parser.add_argument('--use_lr_scheduler', help='Use a polynomial LR scheduler to vary the learning rate', action='store_true')
parser.add_argument('--loss', choices=['bceloss', 'diceloss'], help="Loss function to use for training and validation", type=str, default='diceloss')
parser.add_argument('--batch_size', help='The number of samples per batch', type=int, default=4)
parser.add_argument('--n_epochs', help='Hard limit of maximum number of epochs performed during training', type=int, default=100)
parser.add_argument('--use_transforms', help='Enable data transformations', action='store_true')
parser.add_argument('--n_augmentation', help='The number of pass on the dataset with different transformations perfomed at each training iteration', type=int, default=10) # Set to 1 for only 1 pass
parser.add_argument('--early_stopping_threshold', help='Number of epochs given to the model to improve on previously better result', type=int, default=10)
parser.add_argument('-d', '--debug', help='To enable / disable the show_val_samples routine', action='store_true')
args = parser.parse_args()

MODEL_NAME = 'segformer'
LR = 0.00006
N_WORKERS = 4 # Base is 4, set to 0 if it causes errors

# To create folders for test predictions and model checkpoints
CURR_DATE = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
CHECKPOINTS_FOLDER = f"checkpoints/segformer_{CURR_DATE}"
TENSORBOARD_FOLDER = f"tensorboard/segformer_{CURR_DATE}"
INFERENCE_FOLDER = f"predictions/raw_{CURR_DATE}"

TRAIN_DATASET_PATH = 'dataset/training'
TEST_DATASET_PATH = 'dataset/test/images/'
CHECKPOINTS_FILE_PREFIX = 'epoch'
INFERENCE_FILE_PREFIX = 'satimage'

def main():
    print(f"Using {DEVICE} device")

    if args.single_split_train:
        print(f"Training on a single train/validation split")
    else:
        print(f"K-fold training with k={args.n_folds}")

    #############################
    # Training routine
    #############################
    # Create a directory for model checkpoints
    os.makedirs(CHECKPOINTS_FOLDER, exist_ok=True)

    print("Starting training")
    # Selecting the transformations to perform for data augmentation
    if args.use_transforms:
        transforms = compose.Compose([
            rotation.Rotation(angle=30, probability=0.6),
            colorjitter.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            randomerasing.RandomErasing(probability=0.5, scale=(0.01, 0.2), ratio=(0.3, 3.3))
        ])
    else:
        transforms = None
    
    # Loading the whole training dataset
    images_dataset = ImageDataset(
        for_train=True,
        images_dir = os.path.abspath(f'{TRAIN_DATASET_PATH}/images/'),
        masks_dir = os.path.abspath(f'{TRAIN_DATASET_PATH}/groundtruth/'),
        transforms=transforms,
        use_patches=False,
        # img_size=(512, 512)
    )

    # Instantiating the k-fold splits
    kfold = KFold(n_splits=args.n_folds, shuffle=True)

    best_epochs = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(images_dataset)):
        if args.single_split_train and fold+1 > 1:
            break

        # Create a directory for model checkpoints
        os.makedirs(f"{CHECKPOINTS_FOLDER}/fold_{fold+1}", exist_ok=True)

        # Initializing Tensorboard
        writer = SummaryWriter(log_dir=f"{TENSORBOARD_FOLDER}/fold_{fold+1}")

        # Setting up the model, loss function and optimizer
        model = SegFormer(non_void_labels=['road'], checkpoint='nvidia/mit-b5').to(DEVICE)
        if args.loss == 'diceloss':
            loss_fn = DiceLoss(model=MODEL_NAME)
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        if args.use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer=optimizer, total_iters=args.n_augmentation*100, power=0.9)
        
        train_dataloader = DataLoader(
            dataset=images_dataset,
            batch_size=args.batch_size,
            num_workers=N_WORKERS,
            sampler=SubsetRandomSampler(train_idx)
        )

        validation_dataloader = DataLoader(
            dataset=images_dataset,
            batch_size=args.batch_size,
            num_workers=N_WORKERS,
            sampler=SubsetRandomSampler(val_idx)
        )

        # Early stopping mechanism
        best_epoch = 0
        best_loss = 100

        for epoch in range(args.n_epochs):
            # Perform training
            model.train()
            losses = [] # To record metric
            # Perform data augmentation by re-feeding n times the training dataset with random transformations each time
            print(f"Epoch {epoch+1} / {args.n_epochs}")
            for n_a in range(args.n_augmentation):
                # For the progress bar (and to load the images from the mini-batch)
                progress_bar = tqdm(iterable=train_dataloader, desc=f"Pass {n_a+1} / {args.n_augmentation}")
                for (x, y) in progress_bar: # x = images, y = labels
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)

                    optimizer.zero_grad() # Zero-out gradients
                    y_hat = model(x) # Forward pass
                    loss = loss_fn(y_hat, y)
                    losses.append(loss.item())
                    # Apply a sigmoid to the inference result since we chose SegFormer as a model
                    y_hat = torch.sigmoid(y_hat)
                    loss.backward() # Backward pass
                    optimizer.step()
                if args.use_lr_scheduler:
                    print(f"Current LR: {scheduler.get_last_lr()}")
                    scheduler.step()

            mean_loss = torch.tensor(losses).mean()
            writer.add_scalar("Loss/training", mean_loss.item(), epoch)

            # Perform validation
            model.eval()
            with torch.no_grad():
                # To compute the overall patch accuracy
                batch_patch_acc, batch_iou, batch_precision, batch_recall, batch_f1, batch_patch_f1 = [], [], [], [], [], []

                losses = [] # For the early stopping mechanism
                progress_bar_validation_dataloader = tqdm(iterable=validation_dataloader, desc=f"Performing validation...")
                for (x, y) in progress_bar_validation_dataloader:
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)

                    y_hat = model(x) # Perform forward pass
                    loss = loss_fn(y_hat, y)
                    # Apply a sigmoid to the inference result since we chose SegFormer as a model
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
                    model.save_pretrained(f"{CHECKPOINTS_FOLDER}/fold_{fold+1}/{CHECKPOINTS_FILE_PREFIX}-{best_epoch+1}.pth")
                elif epoch - best_epoch >= args.early_stopping_threshold:
                    best_epochs.append(best_epoch+1)
                    print(f"Early stopped at epoch {epoch+1} with best epoch {best_epoch+1}")
                    print("------------------")
                    break
        
        writer.flush()
        writer.close()

    #############################
    # Inference routine
    #############################
    # Load best models from checkpoints
    n_checkpoints = len(best_epochs)
    models = [model.load_pretrained(checkpoint=f"{CHECKPOINTS_FOLDER}/fold_{i+1}/{CHECKPOINTS_FILE_PREFIX}-{best_epochs[i]}.pth").to(DEVICE) for i in range(n_checkpoints)]

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

    aggregate = None
    for model in models:
        images = []
        model.eval()
        with torch.no_grad():
            img_idx = 144 # Test images start at index 144
            for x, _ in test_dataloader:
                x = x.to(DEVICE)
                pred = model(x).detach().cpu()
                pred = torch.sigmoid(pred)
                # Add channels to end up with RGB tensors, and save the predicted masks on disk
                pred = torch.cat([pred.moveaxis(1, -1)]*3, -1).moveaxis(-1, 1) # Here the dimension 0 is for the number of images, since we feed a batch!
                images.append(pred)
            if aggregate is None:
                aggregate = torch.cat(images, dim=0)
            else:
                aggregate += torch.cat(images, dim=0)
    
    aggregate = aggregate / n_checkpoints

    for t in aggregate:
        t = (t * 255).type(torch.uint8) # Rescale everything in the 0-255 range to generate channels for images
        write_png(input=t, filename=f'{INFERENCE_FOLDER}/{INFERENCE_FILE_PREFIX}_{img_idx}.png')
        img_idx += 1


if __name__ == "__main__":
    main()
