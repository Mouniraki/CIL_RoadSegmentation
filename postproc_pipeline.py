import os
from datetime import datetime # To save the predicted masks in a dated folder
from tqdm import tqdm

# PyTorch imports
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.io import write_png

# For k-fold cross-validation
from sklearn.model_selection import KFold

# Importing the dataset & models
from utils.loaders.image_dataset import ImageDataset
from utils.loaders.transforms import compose, rotation, colorjitter, randomerasing, randomresizedcrop
from utils.models.unet import UNet
from utils.models.segformer import SegFormer
from utils.losses.diceloss import DiceLoss

# Importing plot & metric utilities
from utils.plotting import plot_patches, show_val_samples
from utils.metrics import patch_accuracy_fn, iou_fn, precision_fn, recall_fn, f1_fn, patch_f1_fn

# To select the proper hardware accelerator
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Constants
PATCH_SIZE = 16
CUTOFF = 0.25

SELECTED_MODEL = "segformer" # Set this to the desired model
DEBUG = True # To enable / disable the show_val_samples routine
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

N_AUGMENTATION = 5 # Set to 1 for only 1 pass

# Function for postprocessing experimentations based on the best model trained so far (see description.txt for more information)
def postprocessing_pipeline(folder_name: str = '23-07-2024_14-51-05', loss_type: str = 'diceloss', n_models: int = 5, best_epochs: list[int] = [16, 35, 27, 19, 34]):
    print(f"Using {DEVICE} device")

    checkpoints_folder = f"checkpoints/{folder_name}"

    if len(best_epochs) != n_models:
        return "Error: the array of best models does not match the number of models specified."


    # # Initializing Tensorboard
    # writer = SummaryWriter(log_dir=f"valid_{TENSORBOARD_FOLDER}")
    

    #############################
    # Checkpoints inference routine
    #############################
    # Loading the whole training dataset for postprocessing
    transforms = compose.Compose([rotation.Rotation(angle=30, probability=0.6),
                                colorjitter.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                randomerasing.RandomErasing(probability=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))])

    images_dataset = ImageDataset(
        for_train=True,
        images_dir = os.path.abspath(f'{TRAIN_DATASET_PATH}/images/'),
        masks_dir = os.path.abspath(f'{TRAIN_DATASET_PATH}/groundtruth/'),
        transforms=transforms,
        use_patches=False,
        # img_size=(512, 512)
    )

    eval_loader = DataLoader(
                    dataset=images_dataset,
                    batch_size=BATCH_SIZE,
                    num_workers=N_WORKERS,
    )

    # Doing model selection
    if SELECTED_MODEL == 'segformer':
        model = SegFormer(non_void_labels=['road'], checkpoint='nvidia/mit-b5')
        if loss_type == 'diceloss':
            loss_fn = DiceLoss(model = SELECTED_MODEL)
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        model = UNet()
        if loss_type == 'diceloss':
            loss_fn = DiceLoss(model = SELECTED_MODEL)
        else:
            loss_fn = torch.nn.BCELoss()

    # Loading the best models from checkpoints
    if n_models > 1:
        models = [model.load_pretrained(checkpoint=f"{checkpoints_folder}/fold_{i+1}/{CHECKPOINTS_FILE_PREFIX}-{best_epochs[i]}.pth").to(DEVICE) for i in range(n_models)]
    else:
        models = [model.load_pretrained(checkpoint=f"{checkpoints_folder}/{CHECKPOINTS_FILE_PREFIX}-{best_epochs[0]}.pth").to(DEVICE)]

    # Setting all models to eval mode
    for m in models:
        m.eval()

    with torch.no_grad():
        # For debugging visualization
        val_samples, val_predictions, ground_truths = [], [], []

        # To compute the overall patch accuracy
        batch_patch_acc, batch_iou, batch_precision, batch_recall, batch_f1, batch_patch_f1 = [], [], [], [], [], []
        losses = []

        progress_bar = tqdm(iterable=eval_loader, desc=f"Processing the samples...")
        for (x, y) in progress_bar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_hat = torch.stack([m(x) for m in models]).mean(dim=0)
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
            batch_patch_f1.append(patch_f1_fn(y_hat=y_hat, y=y))

        # Computing the metrics
        mean_loss = torch.tensor(losses).mean()
        patch_acc = torch.cat(batch_patch_acc, dim=0).mean()
        mean_iou = torch.cat(batch_iou, dim=0).mean()
        precision = torch.cat(batch_precision, dim=0).mean()
        recall = torch.cat(batch_recall, dim=0).mean()
        f1 = torch.cat(batch_f1, dim=0).mean()
        patch_f1 = torch.cat(batch_patch_f1, dim=0).mean()

        # writer.add_scalar("Loss/eval", mean_loss, epoch)
        # writer.add_scalar(f"Accuracy/eval", patch_acc, epoch)
        # writer.add_scalar(f"Mean IoU/eval", mean_iou, epoch)
        # writer.add_scalar(f"Precision/eval", precision, epoch)
        # writer.add_scalar(f"Recall/eval", recall, epoch)
        # writer.add_scalar(f"F1 score/eval", f1, epoch)
        # writer.add_scalar(f"Patch F1 score/eval", patch_f1, epoch)

        print(f"Overall patch accuracy: {patch_acc}")
        print(f"Mean IoU: {mean_iou}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Patch F1 Score: {patch_f1}")
        print(f"Loss: {mean_loss}")

        if DEBUG:
            # For debugging purposes : display the validation samples used for validation
            show_val_samples(torch.cat(val_samples, dim=0), torch.cat(ground_truths, dim=0), torch.cat(val_predictions, dim=0))

    # kfold = KFold(n_splits=5, shuffle=True)

    # for fold, (train_idx, val_idx) in enumerate(kfold.split(images_dataset)):
    #     print(f"Fold {fold+1}")
    #     train_loader = DataLoader(
    #             dataset=images_dataset,
    #             batch_size=BATCH_SIZE,
    #             num_workers=N_WORKERS,
    #             sampler=SubsetRandomSampler(train_idx)
    #         )

    #     eval_loader = DataLoader(
    #             dataset=images_dataset,
    #             batch_size=BATCH_SIZE,
    #             num_workers=N_WORKERS,
    #             sampler=SubsetRandomSampler(val_idx)
    #         )

    #     # Doing model selection
    #     if SELECTED_MODEL == 'segformer':
    #         model = SegFormer(non_void_labels=['road'])
    #         # loss_fn = torch.nn.BCEWithLogitsLoss()
    #         loss_fn = DiceLoss(model = SELECTED_MODEL)
    #     else:
    #         model = UNet()
    #         # loss_fn = torch.nn.BCELoss()
    #         loss_fn = DiceLoss(model = SELECTED_MODEL)

    #     # Loading the best models from checkpoints
    #     if n_models > 1:
    #         models = [model.load_pretrained(checkpoint=f"{CHECKPOINTS_FOLDER}/fold_{i+1}/{CHECKPOINTS_FILE_PREFIX}-{best_epochs[i]}.pth").to(DEVICE) for i in range(n_models)]
    #     else:
    #         models = [model.load_pretrained(checkpoint=f"{CHECKPOINTS_FOLDER}/{CHECKPOINTS_FILE_PREFIX}-{best_epochs[0]}.pth").to(DEVICE)]


    #     # Setting all models to eval mode
    #     for m in models:
    #         m.eval()

    #     with torch.no_grad():
    #         # For debugging visualization
    #         val_samples, val_predictions, ground_truths = [], [], []

    #         # To compute the overall patch accuracy
    #         batch_patch_acc, batch_iou, batch_precision, batch_recall, batch_f1, batch_patch_f1 = [], [], [], [], [], []
    #         losses = []
    #         for (x, y) in eval_loader:
    #             x = x.to(DEVICE)
    #             y = y.to(DEVICE)
    #             output = torch.tensor([m(x) for m in models]).mean(dim=0)
    #             loss = loss_fn(output, y)

    #             # Apply a sigmoid to the inference result if we use the BCEWithLogitsLoss for metrics computations
    #             if SELECTED_MODEL == 'segformer':
    #                 y_hat = torch.sigmoid(y_hat)

    #             val_samples.append(x.detach().cpu())
    #             val_predictions.append(y_hat.detach().cpu())
    #             ground_truths.append(y.detach().cpu())

    #             losses.append(loss.item())
    #             batch_patch_acc.append(patch_accuracy_fn(y_hat=y_hat, y=y))
    #             batch_iou.append(iou_fn(y_hat=y_hat, y=y))
    #             batch_precision.append(precision_fn(y_hat=y_hat, y=y))
    #             batch_recall.append(recall_fn(y_hat=y_hat, y=y))
    #             batch_f1.append(f1_fn(y_hat=y_hat, y=y))
    #             batch_patch_f1.append(patch_f1_fn(y_hat=y_hat, y=y))

    #             # Computing the metrics
    #             mean_loss = torch.tensor(losses).mean()
    #             patch_acc = torch.cat(batch_patch_acc, dim=0).mean()
    #             mean_iou = torch.cat(batch_iou, dim=0).mean()
    #             precision = torch.cat(batch_precision, dim=0).mean()
    #             recall = torch.cat(batch_recall, dim=0).mean()
    #             f1 = torch.cat(batch_f1, dim=0).mean()
    #             patch_f1 = torch.cat(batch_patch_f1, dim=0).mean()

    #             # writer.add_scalar("Loss/eval", mean_loss, epoch)
    #             # writer.add_scalar(f"Accuracy/eval", patch_acc, epoch)
    #             # writer.add_scalar(f"Mean IoU/eval", mean_iou, epoch)
    #             # writer.add_scalar(f"Precision/eval", precision, epoch)
    #             # writer.add_scalar(f"Recall/eval", recall, epoch)
    #             # writer.add_scalar(f"F1 score/eval", f1, epoch)
    #             # writer.add_scalar(f"Patch F1 score/eval", patch_f1, epoch)

    #             print(f"Overall patch accuracy: {patch_acc}")
    #             print(f"Mean IoU: {mean_iou}")
    #             print(f"Precision: {precision}")
    #             print(f"Recall: {recall}")
    #             print(f"F1 Score: {f1}")
    #             print(f"Patch F1 Score: {patch_f1}")
    #             print(f"Loss: {mean_loss}")


postprocessing_pipeline()