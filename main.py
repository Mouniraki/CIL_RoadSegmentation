import os
from datetime import datetime # To save the predicted masks in a dated folder
from tqdm import tqdm

# PyTorch imports
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torchvision.io import write_png

# For k-fold cross-validation
from sklearn.model_selection import KFold

# Importing the dataset & models
from utils.loaders.image_dataset import ImageDataset
from utils.loaders.transforms import compose, colorjitter, randomresizedcrop
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

SELECTED_MODEL = "segformer" # Set this to the desired model
DEBUG = False # To enable / disable the show_val_samples routine
K_FOLDS = 5
LR = 0.00006
BATCH_SIZE = 4
N_WORKERS = 4 # Base is 4, set to 0 if it causes errors
N_EPOCHS = 100
EARLY_STOPPING_THRESHOLD = 10

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

    #############################
    # Training routine
    #############################
    print("Starting training")
    # Selecting the transformations to perform for data augmentation
    transforms = compose.Compose([
        colorjitter.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        randomresizedcrop.RandomResizedCrop(scale=(0.5, 1), ratio=(3/4, 4/3))
    ])

    # Instantiating the k-fold splits
    kfold = KFold(n_splits=K_FOLDS, shuffle=True)

    # Loading the whole training dataset
    images_dataset = ImageDataset(
        for_train=True,
        images_dir = os.path.abspath(f'{TRAIN_DATASET_PATH}/images/'),
        masks_dir = os.path.abspath(f'{TRAIN_DATASET_PATH}/groundtruth/'),
        transforms=transforms,
        use_patches=False,
        # img_size=(512, 512)
    )

    best_epochs = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(images_dataset)):
        print(f"Fold {fold+1}")

        # Create a directory for model checkpoints
        os.makedirs(f"{CHECKPOINTS_FOLDER}/fold_{fold+1}", exist_ok=True)

        # Initializing Tensorboard
        writer = SummaryWriter(log_dir=f"{TENSORBOARD_FOLDER}/fold_{fold+1}")

        # Setting up the model, loss function and optimizer
        if SELECTED_MODEL == 'segformer':
            model = SegFormer(non_void_labels=['road'], checkpoint='nvidia/mit-b5').to(DEVICE)
            loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            model = UNet().to(DEVICE)
            loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

        train_dataloader = DataLoader(
            dataset=images_dataset,
            batch_size=BATCH_SIZE,
            num_workers=N_WORKERS,
            sampler=SubsetRandomSampler(train_idx)
        )

        validation_dataloader = DataLoader(
            dataset=images_dataset,
            batch_size=BATCH_SIZE,
            num_workers=N_WORKERS,
            sampler=SubsetRandomSampler(val_idx)
        )

        # Early stopping mechanism
        best_epoch = 0
        best_loss = 100

        for epoch in range(N_EPOCHS):
            # For the progress bar (and to load the images from the mini-batch)
            progress_bar = tqdm(iterable=train_dataloader, desc=f"Epoch {epoch+1} / {N_EPOCHS}")
            # Perform training
            model.train()
            losses = [] # To record metric
            for (x, y) in progress_bar: # x = images, y = labels
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                optimizer.zero_grad() # Zero-out gradients
                y_hat = model(x) # Forward pass
                # y_hat = adjust_contrast(y_hat, contrast_factor=0.5) # Force the predictions to be more contrasted
                loss = loss_fn(y_hat, y)
                losses.append(loss.item())
                if SELECTED_MODEL == 'segformer':
                    y_hat = torch.sigmoid(y_hat)
                loss.backward() # Backward pass
                optimizer.step()
            
            mean_loss = torch.tensor(losses).mean()
            writer.add_scalar("Loss/train", mean_loss, epoch)

            # Perform validation
            model.eval()
            val_samples, val_predictions, ground_truths = [], [], []
            with torch.no_grad():
                # To compute the overall patch accuracy
                batch_patch_acc, batch_iou, batch_precision, batch_recall, batch_f1 = [], [], [], [], []

                losses = [] # For the early stopping mechanism
                for (x, y) in validation_dataloader:
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)
                    y_hat = model(x) # Perform forward pass
                    loss = loss_fn(y_hat, y)
                    # Apply a sigmoid to the inference result if we use the BCEWithLogitsLoss for metrics computations
                    if SELECTED_MODEL == 'segformer':
                        y_hat = torch.sigmoid(y_hat)

                    val_samples.append(x.detach().cpu())
                    val_predictions.append(y_hat.detach().cpu())
                    ground_truths.append(y.detach().cpu())

                    losses.append(loss.item())
                    batch_patch_acc.append(patch_accuracy_fn(y_hat=y_hat, y=y))
                    batch_iou.append(iou_fn(y_hat=y_hat, y=y))
                    batch_precision.append(precision_fn(y_hat=y_hat, y=y))
                    batch_recall.append(recall_fn(y_hat=y_hat, y=y))
                    batch_f1.append(f1_fn(y_hat=y_hat, y=y))
                
                # Computing the metrics
                mean_loss = torch.tensor(losses).mean()
                patch_acc = torch.cat(batch_patch_acc, dim=0).mean()
                mean_iou = torch.cat(batch_iou, dim=0).mean()
                precision = torch.cat(batch_precision, dim=0).mean()
                recall = torch.cat(batch_recall, dim=0).mean()
                f1 = torch.cat(batch_f1, dim=0).mean()

                writer.add_scalar("Loss/eval", mean_loss, epoch)
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
                print(f"Loss: {mean_loss}")

                if DEBUG:
                    # For debugging purposes : display the validation samples used for validation
                    show_val_samples(torch.cat(val_samples, dim=0), torch.cat(ground_truths, dim=0), torch.cat(val_predictions, dim=0))

                if mean_loss <= best_loss:
                    best_loss = mean_loss
                    best_epoch = epoch
                    # Save a checkpoint
                    model.save_pretrained(f"{CHECKPOINTS_FOLDER}/fold_{fold+1}/{CHECKPOINTS_FILE_PREFIX}-{best_epoch+1}.pth")
                    # torch.save(model.state_dict, f"checkpoints/{CURR_DATE}/epoch-{best_epoch+1}.pth")
                elif epoch - best_epoch >= EARLY_STOPPING_THRESHOLD:
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
    models = [model.load_pretrained(checkpoint=f"{CHECKPOINTS_FOLDER}/fold_{i+1}/{CHECKPOINTS_FILE_PREFIX}-{best_epochs[i]}.pth").to(DEVICE) for i in range(K_FOLDS)]

    print("Performing inference")
    test_dataset = ImageDataset(
        for_train=False,
        images_dir = os.path.abspath(TEST_DATASET_PATH),
        # img_size = (512, 512)
    )
    # We don't shuffle to keep the original data ordering
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=False)

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
                if SELECTED_MODEL == 'segformer':
                    pred = torch.sigmoid(pred)
                # Add channels to end up with RGB tensors, and save the predicted masks on disk
                pred = torch.cat([pred.moveaxis(1, -1)]*3, -1).moveaxis(-1, 1) # Here the dimension 0 is for the number of images, since we feed a batch!
                images.append(pred)
            if aggregate is None:
                aggregate = torch.cat(images, dim=0)
            else:
                aggregate += torch.cat(images, dim=0)
    
    aggregate = aggregate / K_FOLDS

    for t in aggregate:
        t = (t * 255).type(torch.uint8) # Rescale everything in the 0-255 range to generate channels for images
        write_png(input=t, filename=f'{INFERENCE_FOLDER}/{INFERENCE_FILE_PREFIX}_{img_idx}.png')
        img_idx += 1

main()