import argparse
import os
from datetime import datetime # To save the predicted masks in a dated folder
from tqdm import tqdm

# PyTorch imports
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torchvision.io import write_png

# Importing the dataset & models
from utils.loaders.image_dataset import ImageDataset
from utils.loaders.transforms import compose, rotation
from utils.models.unet import UNet
from utils.models.segformer import SegFormer
from utils.losses.loss import DiceLoss

# Importing plot & metric utilities
from utils.plotting import show_val_samples, show_only_labels, save_postProcessing_effect
from utils.metrics import patch_accuracy_fn, iou_fn, precision_fn, recall_fn, f1_fn, patch_f1_fn
from utils.post_processing.post_processing import PostProcessing

# To select the proper hardware accelerator
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Constants
PATCH_SIZE = 16
CUTOFF = 0.25

parser = argparse.ArgumentParser(prog='main', description='The file implement the training loop of the postprocessing model for our CIL project implementation')
parser.add_argument('--n_epochs', help='maximum number of epochs performed during training', type=int, default=100)
parser.add_argument('--early_stopping_threshold', help='Nbr of epoch given to the model to improve on previously better result', type=int, default=10)
parser.add_argument('--n_augmentation', help='Nbr of passes of the whole dataset', type=int, default=5)
parser.add_argument('--batch_size', help='The nbr of sample evaluated in parallel', type=int, default=4)
# parser.add_argument('-m', '--model', help='Set this to the desired model', default="segformer")

parser.add_argument('-pa', '--postprocessing_type', choices=['mask_connected_though_border_radius', 'connect_roads', 'connect_all_close_pixels', 'deepnet'], help='Choose the type of desired postprocessing', type=str, default='deepnet')
parser.add_argument('-rt', '--refinement_training', help="Train the postprocessing network from scratch", action='store_true')
parser.add_argument('-d', '--debug', help='To enable / disable the show_val_samples routine ', action='store_true')
args = parser.parse_args()

LR = 0.00006
N_WORKERS = 4 # Base is 4, set to 0 if it causes errors

# To create folders for test predictions and model checkpoints
CURR_DATE = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
CHECKPOINTS_FOLDER = f"checkpoints/postprocessing_{CURR_DATE}"
TENSORBOARD_FOLDER = f"tensorboard/postprocessing_{CURR_DATE}"
INFERENCE_FOLDER = f"predictions/postprocessing_{CURR_DATE}"

TRAIN_DATASET_PATH = 'dataset/training'
TEST_DATASET_PATH = 'dataset/test/images/'
CHECKPOINTS_FILE_PREFIX = 'epoch'
INFERENCE_FILE_PREFIX = 'satimage'

# For the SegFormer checkpoints
SEGFORMER_MODEL_PATH = "checkpoints/segformer"
BEST_EPOCHS = [40, 15, 59, 14, 10]

# Refinement model checkpoints
REFINEMENT_FINETUNED_PATH = "checkpoints/postprocessing/refinement_finetuned.pth"
LOCAL_EVALUATION = False

# Function for postprocessing experimentations based on the best model trained so far (see description.txt for more information)
def postprocessing_pipeline():
    print(f"Using {DEVICE} device")
    
    # Initializing Tensorboard
    writer = SummaryWriter(log_dir=TENSORBOARD_FOLDER)

    #############################
    # Checkpoints inference routine
    #############################
    # Loading the whole training dataset for postprocessing
    transforms = compose.Compose([
        rotation.Rotation(angle=30, probability=0.8)
    ])

    images_dataset = ImageDataset(
        for_train=True,
        images_dir = os.path.abspath(f'{TRAIN_DATASET_PATH}/images/'),
        masks_dir = os.path.abspath(f'{TRAIN_DATASET_PATH}/groundtruth/'),
        transforms=transforms,
        use_patches=False,
        # img_size=(512, 512)
    )

    # Loading the pretrained checkpoints of the SegFormer model
    model = SegFormer(non_void_labels=['road'], checkpoint='nvidia/mit-b5')
    n_models = len(BEST_EPOCHS)
    if n_models > 1:
        models = [model.load_pretrained(checkpoint=f"{SEGFORMER_MODEL_PATH}/fold_{i+1}/{CHECKPOINTS_FILE_PREFIX}-{BEST_EPOCHS[i]}.pth").to(DEVICE) for i in range(n_models)]
    else:
        models = [model.load_pretrained(checkpoint=f"{SEGFORMER_MODEL_PATH}/{CHECKPOINTS_FILE_PREFIX}-{BEST_EPOCHS[0]}.pth").to(DEVICE)]

    # Setting all loaded SegFormer models to eval mode
    for m in models:
        m.eval()

    #############################
    # Training of the refinement UNet model
    #############################
    if args.postprocessing_type == 'deepnet' and args.refinement_training:
        # Initializing the postprocessing model to train it
        refinement_model = UNet(channels=(1, 64, 128, 256, 512, 1024)).to(DEVICE)
        refinement_loss = DiceLoss(model="refinement") #so that the loss does not do sigmoid
        refinement_optimizer = torch.optim.AdamW(refinement_model.parameters(), lr=LR)

        # Create a folder for the checkpoints to use
        os.makedirs(CHECKPOINTS_FOLDER, exist_ok=True)

        print("Start Refinement training")
        train_dataset, val_dataset = random_split(images_dataset, [0.8, 0.2])
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=N_WORKERS, shuffle=True)
        validation_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=N_WORKERS, shuffle=True)

        # Early stopping mechanism
        best_epoch = 0
        best_loss = 100
    
        for epoch in range(args.n_epochs):
            # Setting the model to training mode
            refinement_model.train()

            # For the progress bar (and to load the images from the mini-batch)
            print(f"Epoch {epoch+1} / {args.n_epochs}")
            losses = [] # To record metric
            # Perform data augmentation by re-feeding n times the training dataset with random transformations each time
            for pass_nb in range(args.n_augmentation):
                progress_bar = tqdm(iterable=train_dataloader, desc=f"Pass {pass_nb+1} / {args.n_augmentation}")
                for (x, y) in progress_bar: # x = images, y = labels
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)

                    # First we perform inference on the SegFormer models
                    with torch.no_grad():
                        y_int = torch.stack([m(x) for m in models]).mean(dim=0)
                        y_int = torch.sigmoid(y_int)
                    
                    # Then we train the postprocessing network on the inference of the SegFormer models
                    refinement_optimizer.zero_grad() # Zero-out gradients
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

                    # Perform inference on the validation samples
                    with torch.no_grad():
                        y_int = torch.stack([m(x) for m in models]).mean(dim=0)
                        y_int = torch.sigmoid(y_int)

                    # Perform evaluation on the inference of the SegFormer models
                    y_hat = refinement_model(y_int)
                    loss = refinement_loss(y_hat, y)
                    losses.append(loss.item())

                    # Appending the metrics to arrays to compute the mean on the validation samples
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

                # Early-stopping mechanism
                if mean_loss <= best_loss:
                    best_loss = mean_loss
                    best_epoch = epoch
                    # Save a checkpoint if the best epoch is greater than 10 (avoid saving too much checkpoints)
                    # if epoch >= 5:
                    refinement_model.save_pretrained(f"{CHECKPOINTS_FOLDER}/{CHECKPOINTS_FILE_PREFIX}-{best_epoch+1}.pth")
                elif epoch - best_epoch >= args.early_stopping_threshold:
                    print(f"Early stopped at epoch {epoch+1} with best epoch {best_epoch+1}")
                    break
        print("Refinement training finished")

    """ #  This code can be left uncommented if the user wants to see some validation results of the post-processing performed locally (warning : no guarantee of  truly representative resulty, the data used may have been used during training).
    #############################
    # Evaluation of post processing routines
    #############################
    if(LOCAL_EVALUATION):
        eval_loader = DataLoader(
                        dataset=images_dataset,
                        batch_size=args.batch_size,
                        num_workers=N_WORKERS,
        )
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
                    match args.postprocessing_type:
                        case 'mask_connected_though_border_radius':
                            y_hat_post_processed = postprocessing.mask_connected_though_border_radius(y_hat, downsample=2,
                                                                                                    contact_radius=3,
                                                                                                    threshold_road_not_road=0)
                        case 'connect_roads':
                            y_hat_post_processed = postprocessing.connect_roads(y_hat, downsample=1, max_dist=70,
                                                                                min_group_size=1, threshold_road_not_road=0,
                                                                                fat=6)
                        case 'connect_all_close_pixels':
                            y_hat_post_processed = postprocessing.connect_all_close_pixels(y_hat, downsample=2,
                                                                                        distance_max=7,
                                                                                        threshold_road_not_road=0)
                else:
                    y_hat_post_processed = refinement_model(torch.sigmoid(y_hat))

                # metrics model raw
                loss = loss_fn(y_hat, y)
                y_hat = torch.sigmoid(y_hat)
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
                    y_hat_post_processed = torch.sigmoid(y_hat_post_processed)
                
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

            if args.debug:
                # For debugging purposes : display the validation samples used for validation
                save_postProcessing_effect(torch.cat(val_predictions, dim=0), torch.cat(val_predictions_p, dim=0), torch.cat(ground_truths, dim=0), title_prefix="postProcessing_result/"+'postProcessingResult', segmentation=False)
"""

    #############################
    # Inference routine
    #############################
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
    
    # Load pretrained weights of either the best trained model or the best checkpoint
    if args.postprocessing_type=='deepnet':
        if args.refinement_training:
            refinement_model = refinement_model.load_pretrained(f"{CHECKPOINTS_FOLDER}/epoch-{best_epoch+1}.pth").to(DEVICE)
        else:
            refinement_model = UNet(channels=(1, 64, 128, 256, 512, 1024)).load_pretrained(REFINEMENT_FINETUNED_PATH).to(DEVICE)
        refinement_model.eval()

    with torch.no_grad():
        img_idx = 144 # Test images start at index 144
        for x, _ in test_dataloader:
            x = x.to(DEVICE)
            pred = torch.stack([m(x) for m in models]).mean(dim=0)
            if args.postprocessing_type == 'deepnet':
                pred = refinement_model(torch.sigmoid(pred)).cpu()
            else: # For the manual postprocessing pipelines
                postprocessing = PostProcessing(postprocessing_patch_size=16, device=DEVICE)
                match args.postprocessing_type:
                    case 'mask_connected_though_border_radius':
                        pred = postprocessing.mask_connected_though_border_radius(mask_connect_roads=pred, 
                                                                                  downsample=1,
                                                                                  contact_radius=3,
                                                                                  threshold_road_not_road=0).cpu()

                    case 'connect_roads':
                        pred = postprocessing.connect_roads(mask_connect_roads=pred,
                                                            downsample=1, 
                                                            max_dist=70,
                                                            min_group_size=1, 
                                                            threshold_road_not_road=0,
                                                            fat=6).cpu()

                    case 'connect_all_close_pixels':
                        pred = postprocessing.connect_all_close_pixels(mask_connect_roads=pred, 
                                                                   downsample=8,
                                                                   distance_max=6,
                                                                   threshold_road_not_road=0)
                        pred = postprocessing.blurring_averaging(pred, kernel_size=7).cpu()
                pred = torch.sigmoid(pred)

            # Add channels to end up with RGB tensors, and save the predicted masks on disk
            pred = torch.cat([pred.moveaxis(1, -1)]*3, -1).moveaxis(-1, 1) # Here the dimension 0 is for the number of images, since we feed a batch!
            for t in pred:
                t = (t * 255).type(torch.uint8) # Rescale everything in the 0-255 range to generate channels for images
                write_png(input=t, filename=f'{INFERENCE_FOLDER}/{INFERENCE_FILE_PREFIX}_{img_idx}.png')
                img_idx += 1

if __name__ == "__main__":
    postprocessing_pipeline()