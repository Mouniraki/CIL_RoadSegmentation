import os
from datetime import datetime # To save the predicted masks in a dated folder
from tqdm import tqdm

# PyTorch imports
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
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

#Refinement model
REFINEMENT = False
PRETRAIN_REFINEMENT_PATH = "utils/models/refinement_weights2"

# Function for postprocessing experimentations based on the best model trained so far (see description.txt for more information)
def postprocessing_pipeline(folder_name: str = '23-07-2024_14-51-05', loss_type: str = 'diceloss', n_models: int = 5, best_epochs: list[int] = [16, 35, 27, 19, 34]):
    print(f"Using {DEVICE} device")

    checkpoints_folder = f"checkpoints/{folder_name}"

    if len(best_epochs) != n_models:
        return "Error: the array of best models does not match the number of models specified."


    # # Initializing Tensorboard
    writer = SummaryWriter(log_dir=f"valid_{TENSORBOARD_FOLDER}")
    

    #############################
    # Checkpoints inference routine
    #############################
    # Loading the whole training dataset for postprocessing
    transforms = compose.Compose([rotation.Rotation(angle=30, probability=0.6)])

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

        if REFINEMENT:
            refinement_model = UNet(channels=(1, 64, 128, 256, 512, 1024))
            refinement_model.load_state_dict(torch.load(PRETRAIN_REFINEMENT_PATH))
            refinement_loss = DiceLoss(model = "refinement") #so that the loss does not do sigmoid
            refinement_optimizer = torch.optim.AdamW(refinement_model.parameters(), lr=LR)
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


    #Refinement model fine-tuning
    if REFINEMENT:
        print("Start Refinement training")
        train_dataset, val_dataset = random_split(images_dataset, [0.8, 0.2])
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=True)
        validation_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=True) 

        # Early stopping mechanism
        best_epoch = 0
        best_loss = 100
    
        for epoch in range(N_EPOCHS):
        # For the progress bar (and to load the images from the mini-batch)
            progress_bar = tqdm(iterable=train_dataloader, desc=f"Epoch {epoch+1} / {N_EPOCHS}")
            refinement_model.train()
            losses = [] # To record metric
            # Perform data augmentation by re-feeding n times the training dataset with random transformations each time
            for _ in range(N_AUGMENTATION):
                for (x, y) in progress_bar: # x = images, y = labels
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)
                    refinement_optimizer.zero_grad() # Zero-out gradients
                    
                    y_int = torch.stack([m(x) for m in models]).mean(dim=0)
                    y_int = torch.sigmoid(y_int)
                    y_hat = refinement_model(y_int)
                    loss = refinement_loss(y_hat, y)
                    losses.append(loss.item())
                    loss.backward() # Backward pass
                    refinement_optimizer.step()
        
            mean_loss = torch.tensor(losses).mean()
            writer.add_scalar("Loss/train", mean_loss.item(), epoch)

    
            refinement_model.eval()
            with torch.no_grad():
                # To compute the overall patch accuracy
                batch_patch_acc, batch_iou, batch_precision, batch_recall, batch_f1, batch_patch_f1 = [], [], [], [], [], []

                losses = [] # For the early stopping mechanism
                for (x, y) in validation_dataloader:
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)
                    
                    y_int = torch.stack([m(x) for m in models]).mean(dim=0)
                    y_int = torch.sigmoid(y_int)
                    y_hat = refinement_model(y_int)
                    loss = refinement_loss(y_hat, y)

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
                if DEBUG:
                    show_val_samples(x.detach().cpu(), y.detach().cpu(), y_hat.detach().cpu())

                if mean_loss <= best_loss:
                    best_loss = mean_loss
                    best_epoch = epoch
                    # Save a checkpoint if the best epoch is greater than 10 (avoid saving too much checkpoints)
                    # if epoch >= 5:
                    model.save_pretrained(f"{CHECKPOINTS_FOLDER}/{CHECKPOINTS_FILE_PREFIX}-{best_epoch+1}.pth")
                    # torch.save(model.state_dict, f"checkpoints/{CURR_DATE}/epoch-{best_epoch+1}.pth")
                elif epoch - best_epoch >= EARLY_STOPPING_THRESHOLD:
                    print(f"Early stopped at epoch {epoch+1} with best epoch {best_epoch+1}")
                    break
    
    print("Refinement training finished")
    # TODO SAVE WEIGHTS
    
    refinement_model.eval()
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
            if not REFINEMENT:
                postprocessing = PostProcessing(postprocessing_patch_size=16)
                y_hat_post_processed = postprocessing.connect_roads(y_hat, downsample=1, max_dist=25, min_group_size=1, threshold_road_not_road=0)
                #y_hat_post_processed = postprocessing.mask_connected_though_border_radius(y_hat, downsample=2, contact_radius=3, threshold_road_not_road=0)
                #y_hat_post_processed = postprocessing.blurring_and_threshold(y_hat, kernel_size=7)
            else:
                y_hat_post_processed = refinement_model(torch.sigmoid(y_hat))
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
            if REFINEMENT:
                loss_post_processed = refinement_loss(y_hat_post_processed, y)
            else:
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
            show_only_labels(torch.cat(val_predictions, dim=0)[5:10], torch.cat(val_predictions_p, dim=0)[5:10], torch.cat(ground_truths, dim=0)[5:10])
            show_only_labels(torch.cat(val_predictions, dim=0)[10:15], torch.cat(val_predictions_p, dim=0)[10:15], torch.cat(ground_truths, dim=0)[10:15])
            show_only_labels(torch.cat(val_predictions, dim=0)[15:20], torch.cat(val_predictions_p, dim=0)[15:20], torch.cat(ground_truths, dim=0)[15:20])
            show_only_labels(torch.cat(val_predictions, dim=0)[20:25], torch.cat(val_predictions_p, dim=0)[20:25], torch.cat(ground_truths, dim=0)[20:25])
    
    writer.flush()
    writer.close()
if __name__ == "__main__":
    postprocessing_pipeline(folder_name="22-07-2024_16-50-49", loss_type="diceloss", n_models=1, best_epochs=[14])