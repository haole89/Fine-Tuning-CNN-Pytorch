import cv2
from . import config
import numpy as np
import os
import shutil
from imutils import paths


def preprocess_image(image):
    # swap the color channels from BGR to RGB, resize it, and scale
    # the pixel values to [0, 1] range
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # (height, width, channel) (224,224,3)
    
    image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    image = image.astype("float32") / 255.0
    # subtract ImageNet mean, divide by ImageNet standard deviation	
    image -= config.MEAN
    image /= config.STD
    # set "channels first" ordering
    # to convert from (224,224,3) to (3,224,224) (channel, height, width)
    # in order to fit with Pytorch
    image = np.transpose(image, (2, 0, 1))
    # add a batch dimension to fit with Pytorch (batch_size, channels, height, width)
    image = np.expand_dims(image, 0)
    # image shape (1,3,224,224)
    # print(image.shape)
    return image


def copy_images(path_lst, folder):
    # check if the destination folder exists and if not create it
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # loop for copy
    for path in path_lst:
        # grab image name and its label from the path and create
        # a placeholder corresponding to the separate label folder
        # structure: flower_photos\img_class\img_name
        img_name = path.split(os.path.sep)[-1]
        label = path.split(os.path.sep)[1]
        label_des = os.path.join(folder, label)
        # check to see if the label folder exists and if not create it
        if not os.path.exists(label_des):
            os.makedirs(label_des)
        # construct the destination image path and copy
        destination = os.path.join(label_des, img_name)
        shutil.copy(path, destination)


