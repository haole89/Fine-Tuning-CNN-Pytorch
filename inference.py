from chip import config
from chip import create_dataloaders
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
import torch

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    # build our data pre-processing pipeline
    test_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    # calculate the inverse mean and standard deviation
    inv_mean = [-m/s for (m, s) in zip(config.MEAN, config.STD)]
    inv_std = [1/s for s in config.STD]
    # define our de-normalization transform
    # to display the output images to our screen
    denormalize = transforms.Normalize(mean=inv_mean, std=inv_std)

    (test_ds, test_loader) = create_dataloaders.get_dataloader(
        config.VAL,	
        transforms= test_transform, 
        batch_size = config.PRED_BATCH_SIZE, shuffle=True
    )

    # check if we have a GPU available, if so, define the map location accordingly
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    # otherwise, we will be using CPU to run our model
    else:
        map_location = "cpu"
    # load the model
    model_path = r"checkpoints\warmup_model.pth"
    model = torch.load(model_path, map_location=map_location)
    # move the model to the device and set it in evaluation mode
    model.to(config.DEVICE)
    model.eval()

    # grab a batch of test data
    batch = next(iter(test_loader))
    (images, labels) = (batch[0].to(config.DEVICE), batch[1].to(config.DEVICE))
    # initialize a figure
    fig = plt.figure("Results", figsize=(10, 10))
    preds = model(images)

    # loop over all the batch
    for i in range(0, config.PRED_BATCH_SIZE):
        # initalize a subplot
        ax = plt.subplot(config.PRED_BATCH_SIZE, 1, i + 1)
        # grab the image, de-normalize it, scale the raw pixel
        # intensities to the range [0, 255], and change the channel
        # ordering from channels first tp channels last
        image = images[i]
        image = denormalize(image).cpu().numpy()
        image = (image * 255).astype("uint8")
        image = image.transpose((1, 2, 0))
        # grab the ground truth label
        idx = labels[i].cpu().numpy()
        gt_label = test_ds.classes[idx]
        # grab the predicted label
        pred = preds[i].argmax().cpu().numpy()
        pred_label = test_ds.classes[pred]
        # add the results and image to the plot
        info = "Ground Truth: {}, Predicted: {}".format(gt_label, pred_label)
        plt.imshow(image)
        plt.title(info)
        plt.axis("off")

    # show the plot
    plt.tight_layout()
    plt.show()



    return

if __name__ == '__main__':
    main()
    