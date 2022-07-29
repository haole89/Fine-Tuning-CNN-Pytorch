import torch
import os

# image dimension
IMAGE_SIZE = 224
# specify Imagenet mean and standard deviation for the RGB image.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# cpu and gpu selection in pytorch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# the label path
IN_LABELS = "ilsvrc2012_wordnet_lemmas.txt"


# define path to the original dataset and base path to the dataset splits
DATA_PATH = "flower_photos"
BASE_PATH = "dataset"

# define validation split and paths to separate train and validation splits
# 90% of the data for training and 10% for validation.
VAL_SPLIT = 0.1
TRAIN = os.path.join(BASE_PATH, "train")
VAL = os.path.join(BASE_PATH, "val")

# specify training hyperparameters
FEATURE_EXTRACTION_BATCH_SIZE = 4
FINETUNE_BATCH_SIZE = 4
PRED_BATCH_SIZE = 10
EPOCHS = 3
LR = 0.001
LR_FINETUNE = 0.0005

# define paths to store training plots and trained model
WARMUP_PLOT = os.path.join("output", "warmup.png")
FINETUNE_PLOT = os.path.join("output", "finetune.png")
WARMUP_MODEL = os.path.join("checkpoints", "warmup_model.pth")
FINETUNE_MODEL = os.path.join("checkpoints", "finetune_model.pth")
