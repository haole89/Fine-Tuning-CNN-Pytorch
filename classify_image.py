from chip import config
from chip.util import preprocess_image
import torch
from torchvision import models
import numpy as np
import cv2

def main():
    print("[INFO] STARTING ...")
    # load VGG16 and put it on GPU
    # model = models.vgg16(pretrained = True)

    model_path = r"checkpoints\\vgg16.pth"
    model = models.vgg16()
    model.load_state_dict(torch.load(model_path))
    model.to(config.DEVICE)
    # to avoid the update weigth
    model.eval()

    # load image
    img_path = r"images\\n01440764_tench.jpg"
    image = cv2.imread(img_path)
    orig = image.copy()
    image = preprocess_image(image)

    # convert the preprocessed image to a torch tensor 
    # and flash it to the current device
    image = torch.from_numpy(image)
    image = image.to(config.DEVICE)

    # load the preprocessed the ImageNet labels
    labels = dict(enumerate(open(config.IN_LABELS)))
    # print(labels)

    # classify the image and extract the predictions   
    logits = model(image)
    # torch.Size([1, 1000])
    
    probabilities = torch.nn.Softmax(dim=-1)(logits)    
    sorted_prob = torch.argsort(probabilities, dim=-1, descending=True)
    # loop over the predictions and display the rank-5 predictions and
    # corresponding probabilities 
    for (i, idx) in enumerate(sorted_prob[0, :5]):
        print("{}. {}: {:.2f}%".format
            (i, labels[idx.item()].strip(),
            probabilities[0, idx.item()] * 100))

    # draw the top prediction on the image and display the image
    (label, prob) = (labels[probabilities.argmax().item()], probabilities.max().item())

    cv2.putText(orig, "Label: {}, {:.2f}%".format(label.strip(), prob * 100),
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imshow("demo", orig)
    cv2.waitKey(0)

    print("[INFO] STOPING ...")
    return

if __name__ == '__main__':
    main()
    