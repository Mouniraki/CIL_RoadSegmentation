import os
from datetime import datetime # To save the predicted masks in a dated folder
from tqdm import tqdm

# PyTorch imports
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torchvision.io import write_png
from torchvision.transforms.v2 import Resize

# Importing the dataset & models
from utils.loaders.image_dataset import ImageDataset
from utils.models.unet import UNet
from transformers import SegformerConfig, SegformerForSemanticSegmentation

# Importing plot & metric utilities
from utils.plotting import plot_patches, show_val_samples
from utils.metrics import patch_accuracy_fn

# To select the proper hardware accelerator
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Constants
PATCH_SIZE = 16
CUTOFF = 0.25

TRAIN_SPLIT = 0.8
BATCH_SIZE = 4
N_WORKERS = 0 # Base is 4, set to 0 if it causes errors
N_EPOCHS = 5

def main():
    print(f"Using {DEVICE} device")

    # Initializing Tensorboard
    writer = SummaryWriter(log_dir='tensorboard/')

    # Setting up the model, loss function and optimizer
    # model = UNet().to(DEVICE)
    id2label = {0: 'road'}
    id2color = {0: [255, 255, 255]} # Road is completely white in our case
    label2id = {'road': 0}
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/mit-b0', 
                                                             num_labels=1, 
                                                             id2label=id2label, 
                                                             label2id=label2id)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)

    #############################
    # Training routine
    #############################
    print("Starting training")
    # Loading the whole training dataset
    images_dataset = ImageDataset(
        for_train=True,
        images_dir = os.path.abspath('dataset/training/images/'),
        masks_dir = os.path.abspath('dataset/training/groundtruth/'),
        use_patches=False,
        img_size=(512, 512)
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

            # TODO: MODIFY THIS TO MAKE IT GENERALIZABLE
            y_hat = torch.sigmoid(y_hat.logits)

            loss = loss_fn(y_hat, y)
            writer.add_scalar("Loss/train", loss.item(), epoch)
            loss.backward() # Backward pass
            optimizer.step()
        
        # Perform validation
        model.eval()
        with torch.no_grad():
            # To compute the overall patch accuracy
            images, predictions, ground_truths = [], [], []
            for (x, y) in validation_dataloader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                y_hat = model(x) # Perform forward pass
                y_hat = torch.sigmoid(y_hat.logits)
                loss = loss_fn(y_hat, y)
                writer.add_scalar("Loss/eval", loss.item(), epoch)
                images.append(x)
                predictions.append(y_hat)
                ground_truths.append(y)
            
            images = torch.cat(images, 0)
            predictions = torch.cat(predictions, 0)
            ground_truths = torch.cat(ground_truths, 0)
            patch_acc = patch_accuracy_fn(ground_truths, predictions, patch_size=PATCH_SIZE, cutoff=CUTOFF)
            writer.add_scalar(f"Accuracy/eval", patch_acc, epoch)
            print(f"Overall patch accuracy: {patch_acc}")
        
        # Optional : display the validation samples used for validation
        show_val_samples(images.detach().cpu(), ground_truths.detach().cpu(), predictions.detach().cpu())


    #############################
    # Testing routine
    #############################
    print("Performing testing")
    test_dataset = ImageDataset(
        for_train=False,
        images_dir = os.path.abspath('dataset/test/images/'),
        img_size = (512, 512))
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
            pred = model(x)
            pred = torch.sigmoid(pred.logits.detach().cpu())
            # Add channels to end up with RGB tensors, and save the predicted masks on disk
            pred = torch.cat([pred.moveaxis(1, -1)]*3, -1).moveaxis(-1, 1) # Here the dimension 0 is for the number of images, since we feed a batch!
            for t in pred:
                t = (t * 255).type(torch.uint8) # Rescale everything in the 0-255 range to generate channels for images
                pred = Resize(size=(400, 400)).forward(pred) # TODO: MAKE THIS GENERALIZABLE
                write_png(input=t, filename=f'predictions/{curr_date}/satimage_{img_idx}.png')
                img_idx += 1

    writer.flush()
    writer.close()

main()