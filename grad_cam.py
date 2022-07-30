import torch
from torchvision.models import vgg16
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
from model.gradcam import GradCam

def main():

    IMAGE_NAME = r"images\\n01440764_tench.jpg"
    SAVE_NAME = r"output\\grad_cam.jpg"
    test_image = (transforms.ToTensor()(Image.open(IMAGE_NAME))).unsqueeze(dim=0)
    orignal_model = vgg16()
    model_path = r"checkpoints\\vgg16.pth"
    orignal_model.load_state_dict(torch.load(model_path))

    grad_cam = GradCam(orignal_model)
    feature_image = grad_cam(test_image).squeeze(dim=0)
    feature_image = transforms.ToPILImage()(feature_image)
    feature_image.save(SAVE_NAME)
    
    return

if __name__ == '__main__':
    main()
    