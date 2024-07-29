# ETHZ CIL Road Segmentation 2024 (Team : Highway Hackers)

[CIL Road Segmentation 2024 : report Mounir Raki, Théo Ducrey, Marchon Xavier ](report.pdf)

**Initial Code Release:** This directory currently provides our implementation of our model segformer and optimization as described in our report for the project of road Segmentation. This work was done in the context of the course Computational Intelligence Lab at ETH Zürich.

File structure of principal parts:      

```
├── checkpoints
│ └── 23-07-2024_14-51-05           -> Contain the weights of our best model
│ │ └── description.txt             -> describe the parameters used in training for each specific checkpoints
├── dataset                         -> Can be downloaded from TODO
│ ├── test
│ │ └── images
│ └── training
│     ├── groundtruth
│     └── images
├── job-run.sh                      -> A simple job that can be used to run the pipeline on the cluster if needed
├── main.py                         -> File containing the necessary code to train our model and perform inference on the test data without any postprocessing
├── postproc_pipeline.py            -> File containing the necessary code to load a checkpoint run evaluation and inference on it and postprocess each predicted mask using different available algorithms
├── README.md                       -> Instruction on how to install and run basic example including explanation on the content of the directory
└── utils
    ├── env_setup                   
    │ ├── job-install.sh            -> Simple job allowing to setup an environment for our project in a cluster environment
    │ ├── move.py                   -> Python code prototype that can be used to convert an external dataset to the format used in this project
    │ └── requirements.txt          -> Library and version used in the project 
    ├── loaders
    │ ├── image_dataset.py          -> Code to load a dataset of test and train image pair, this code was given to us by the CIL teaching team
    │ └── transforms
    │     ├── colorjitter.py        -> Class used to introduce random color changes in the training images (Data augmentation)
    │     ├── compose.py            -> Class used to compose a series of transformations on the training images (Data augmentation)
    │     ├── randomerasing.py      -> Class used to erase random patch of the training images (Data augmentation)
    │     ├── randomresizedcrop.py  -> Class used to resize and crop randomly the training images (Data augmentation)
    │     └── rotation.py           -> Class used to rotate randomly the training images (Data augmentation)
    ├── losses
    │ ├── loss.py                   -> Implements the losses used in the training loop
    ├── metrics.py                  -> Implement the method used to compute the metrics : patch_accuracy_fn / precision_fn / recall_fn / f1_fn / patch_f1_fn /                                             iou_fn 
    ├── models
    │ ├── segformer.py              -> Define our model with the configuration adapted to the task of road segmentation
    │ └── unet.py                   -> Define a unet model, used in our initial testing based on the code furnished by the teaching team and used in the deep postprocessing model
    ├── plotting.py                 -> Define the utils method used for displaying and saving the resulting mask and images (plot_patches / show_val_samples were given by the teaching team)
    ├── post_processing
    │ ├── post_processing.py        -> Implementation of the method of postprocessing as defined in the report
    └── submission
        ├── mask_to_submission.py   -> Code given by Kaggle to convert our prediction to the format accepted by the platfrom
        └── submission_to_mask.py   -> Code reverting a Kaggle submission to images
```

## Requirements

* **Inference and training were performed with a GPU of 8GB of memory. In case more memory is availible, you may consider increasing the batch_size to increase performance.

## Getting Started
1. Download the checkpoints and the dataset on polybox
```
dataset -> CIL_RoadSegmentation
checkpoints -> CIL_RoadSegmentation
finetuned3 -> utils/models
```
2. Install dependencies

Create and activate a virtual environment
```
python3 -m venv env_cil
source env_cil/bin/activate
pip install -r env_setup/requirements.txt
```



## Run
### Training
```
python3 main.py --n_epochs=100 --n_augmentation=4 --early_stopping_threshold=10 --batch_size=4 --debug=True --model="segformer"
```

### PostProcessing / Inference
```
python3 postproc_pipeline.py --n_epochs=100 --early_stopping_threshold=10 --batch_size=4 --debug=True --model="segformer"
```
