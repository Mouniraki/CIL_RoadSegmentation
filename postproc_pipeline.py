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
from utils.plotting import plot_patches, show_val_samples, show_only_labels
from utils.metrics import patch_accuracy_fn, iou_fn, precision_fn, recall_fn, f1_fn, patch_f1_fn
from utils.post_processing.post_processing import PostProcessing

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
        val_samples, ground_truths, val_predictions, val_predictions_p = [], [], [], []

        # To compute the overall patch accuracy
        batch_patch_acc, batch_iou, batch_precision, batch_recall, batch_f1, batch_patch_f1, losses = [], [], [], [], [], [], []
        batch_patch_acc_p, batch_iou_p, batch_precision_p, batch_recall_p, batch_f1_p, batch_patch_f1_p, losses_p = [], [], [], [], [], [], []

        progress_bar = tqdm(iterable=eval_loader, desc=f"Processing the samples...")
        for (x, y) in progress_bar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            val_samples.append(x.detach().cpu())
            ground_truths.append(y.detach().cpu())

            #compute model prediction
            y_hat = torch.stack([m(x) for m in models]).mean(dim=0)



            #pass the prediction though the postprocessing module
            postprocessing = PostProcessing(postprocessing_patch_size=16)
            y_hat_post_processed = postprocessing.connect_roads(y_hat, downsample=1, max_dist=25, min_group_size=1, threshold_road_not_road=0)
            #y_hat_post_processed = postprocessing.mask_connected_though_border_radius(y_hat, downsample=2, contact_radius=3, threshold_road_not_road=0)


            # metrics model raw
            loss = loss_fn(y_hat, y)
            if SELECTED_MODEL == 'segformer': y_hat = torch.sigmoid(y_hat)
            val_predictions.append(y_hat.detach().cpu())
            losses.append(loss.item())
            batch_patch_acc.append(patch_accuracy_fn(y_hat=y_hat, y=y))
            batch_iou.append(iou_fn(y_hat=y_hat, y=y))
            batch_precision.append(precision_fn(y_hat=y_hat, y=y))
            batch_recall.append(recall_fn(y_hat=y_hat, y=y))
            batch_f1.append(f1_fn(y_hat=y_hat, y=y))
            batch_patch_f1.append(patch_f1_fn(y_hat=y_hat, y=y))

            # metrics model postprocessed
            loss_post_processed = loss_fn(y_hat_post_processed, y)
            if SELECTED_MODEL == 'segformer': y_hat_post_processed = torch.sigmoid(y_hat_post_processed)
            val_predictions_p.append(y_hat_post_processed.detach().cpu())
            losses_p.append(loss_post_processed.item())
            batch_patch_acc_p.append(patch_accuracy_fn(y_hat=y_hat_post_processed, y=y))
            batch_iou_p.append(iou_fn(y_hat=y_hat_post_processed, y=y))
            batch_precision_p.append(precision_fn(y_hat=y_hat_post_processed, y=y))
            batch_recall_p.append(recall_fn(y_hat=y_hat_post_processed, y=y))
            batch_f1_p.append(f1_fn(y_hat=y_hat_post_processed, y=y))
            batch_patch_f1_p.append(patch_f1_fn(y_hat=y_hat_post_processed, y=y))


        print("----------Model raw metrics----------")
        print(f"Overall patch accuracy: {torch.cat(batch_patch_acc, dim=0).mean()}")
        print(f"Mean IoU: { torch.cat(batch_iou, dim=0).mean()}")
        print(f"Precision: {torch.cat(batch_precision, dim=0).mean()}")
        print(f"Recall: {torch.cat(batch_recall, dim=0).mean()}")
        print(f"F1 Score: {torch.cat(batch_f1, dim=0).mean()}")
        print(f"Patch F1 Score: {torch.cat(batch_patch_f1, dim=0).mean()}")
        print(f"Loss: {torch.tensor(losses).mean()}")
        print("----------Model postprocessed metrics:----------")
        print(f"Overall patch accuracy: {torch.cat(batch_patch_acc_p, dim=0).mean()}")
        print(f"Mean IoU: { torch.cat(batch_iou_p, dim=0).mean()}")
        print(f"Precision: {torch.cat(batch_precision_p, dim=0).mean()}")
        print(f"Recall: {torch.cat(batch_recall_p, dim=0).mean()}")
        print(f"F1 Score: {torch.cat(batch_f1_p, dim=0).mean()}")
        print(f"Patch F1 Score: {torch.cat(batch_patch_f1_p, dim=0).mean()}")
        print(f"Loss: {torch.tensor(losses_p).mean()}")

        if DEBUG:
            # For debugging purposes : display the validation samples used for validation
            show_val_samples(torch.cat(val_samples, dim=0), torch.cat(ground_truths, dim=0), torch.cat(val_predictions, dim=0))
            show_val_samples(torch.cat(val_samples, dim=0), torch.cat(ground_truths, dim=0), torch.cat(val_predictions_p, dim=0))
            show_only_labels(torch.cat(val_predictions, dim=0), torch.cat(val_predictions_p, dim=0), torch.cat(ground_truths, dim=0))

if __name__ == "__main__":
    postprocessing_pipeline()