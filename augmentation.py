import math
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from random import sample
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import Augmentor



"""
Pipeline data augmentation creation
"""
#Transformations that need also transformation of background image
def augmentation_pipeline():
    p = Augmentor.Pipeline(source_directory="training/images", save_format="png")
    p.ground_truth("training/groundtruth")
    p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)
    p.flip_top_bottom(probability=0.5)
    p.flip_left_right(probability=0.5)
    p.shear(probability=0.3, max_shear_left=20, max_shear_right=20)
    p.random_distortion(probability=0.3, grid_width=4, grid_height=4, magnitude=8)
    return p

#Transformations that don't need transformation of background image
#TODO a voir si utile
def augmentation_pipeline_no_groundtruth():
    p.random_erasing(probability=0.5, rectangle_area=0.3)



"""
Image number choice and run the pipeline
"""
def run_augmentation_pipeline(number_of_images):
    p = augmentation_pipeline()
    p.sample(number_of_images)