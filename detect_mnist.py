import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import random
import time
import tensorflow as tf
from yolo_core.models import Create_Yolo
from yolo_core.utils import detect_image
from configuration import *

n_imgs = 20

yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, DATA_TYPE)
yolo.load_weights(save_directory)

"""
# Detection test for mnist
for idx in range(n_imgs):
    ID = random.randint(0, 1000)
    label_txt = "./dataset/mnist/mnist_val.txt"
    image_info = open(label_txt).readlines()[ID].split()
    
    img_name = image_info[0].split("/")

    image_path = image_info[0]

    detect_image(yolo, image_path, "./pred_IMAGES/mnist/Pred_"+img_name[4], input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
"""
# Detection test for fashion mnist
for idx in range(n_imgs):
    ID = random.randint(0, 1000)
    label_txt = "./dataset/fashion_mnist/mnist_val.txt"
    image_info = open(label_txt).readlines()[ID].split()
    
    image_path = image_info[0]
    img_name = image_info[0].split("/")

    detect_image(yolo, image_path, "./pred_IMAGES/fashion_mnist/Pred_"+img_name[4], input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))


