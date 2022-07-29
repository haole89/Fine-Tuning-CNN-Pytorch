from chip import config
from chip.util import copy_images
from imutils import paths
import numpy as np
import os

# paths: A submodule of imutils used to gather paths to images inside a given directory
# shutil: Used to copy files from one location to another

def main():
    # load all the image paths and randomly shuffle them
    print("[INFO] loading image paths...")
    img_path_lst = list(paths.list_images(config.DATA_PATH))
    np.random.shuffle(img_path_lst)

    # generate training and validation paths
    val_path_len = int(len(img_path_lst) * config.VAL_SPLIT)
    train_path_Len = len(img_path_lst) - val_path_len

    train_path_lst = img_path_lst[:train_path_Len]
    val_path_lst = img_path_lst[train_path_Len:]
    # copy the training and validation images to their respective directories
    print("[INFO] copying training and validation images...")
    copy_images(train_path_lst, config.TRAIN)
    copy_images(val_path_lst, config.VAL)

    return

if __name__ == '__main__':
    main()
    