from chip import config
from chip import create_dataloaders
from imutils import paths
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchsummary import summary
from model.finetune import FineTuneVGG16

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_pretrained_model(model_path, num_classes):

    # # load VGG model 
    model = models.vgg16()
    # print(model)
    model.load_state_dict(torch.load(model_path))
    # model = models.vgg16(pretrained = True)

    # since we are using the pre-trained model as a feature extractor we set
    # its parameters to non-trainable (do not calculate gradient to update weights)
    for param in model.parameters():
        param.requires_grad = False

    # append a new classification top to our feature extractor and pop it
    # on to the current device
    # Note: important to remember the name of layers of CNN model
    # The lasted layer of VGG is classifier[0]

    # print(model)
    last_item_index = len(model.classifier) - 1
    old_fc = model.classifier.__getitem__(last_item_index)
    new_fc = nn.Linear(in_features=old_fc.in_features, out_features= num_classes, bias=True)
    model.classifier.__setitem__(last_item_index , new_fc)

    # define the network head and attach it to the model
    # n_inputs = model.classifier[0].in_features
    # head_model = nn.Sequential(
    #     nn.Linear(n_inputs, 4096),
    #     nn.ReLU(),
    #     nn.Dropout(p=0.5),
    #     nn.Linear(in_features=4096, out_features=4096),
    #     nn.ReLU(),
    #     nn.Dropout(p=0.5),
    #     nn.Linear(in_features=4096, out_features=num_classes) 
    # )
    # model.classifier = head_model

    print(model)
    model = model.to(config.DEVICE)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad is True.

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    return model

# define augmentation pipelines
# Randomly perform rotation by in the range [-90, 90]
def main():    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])

    # Notice that we do not perform data augmentation inside the validation transformer
    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    # create data loaders
    (train_ds, train_loader) = create_dataloaders.get_dataloader(
        config.TRAIN,
        transforms=train_transform,
        batch_size=config.FEATURE_EXTRACTION_BATCH_SIZE
        )

    (val_ds, val_loader) = create_dataloaders.get_dataloader(    
        config.VAL,
        transforms=val_transform,
        batch_size=config.FEATURE_EXTRACTION_BATCH_SIZE, shuffle=False
        )
    print("Length of the train / val set:", len(train_ds), "/", len(val_ds))

    # get pretrained model
    num_classes = len(train_ds.classes)
    model_path = r"checkpoints\\vgg16.pth"

    # model_ft = get_pretrained_model(model_path, num_classes)
    # summary(model_ft)
    # print(model_ft.fc)

    model_ft = FineTuneVGG16(model_path=model_path, num_classes=num_classes)
    print(model_ft)
    model_ft.to(config.DEVICE)

    total_params = sum(p.numel() for p in model_ft.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model_ft.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    # to train we initialize our loss function and optimization method
    # initialize loss function and optimizer (notice that we are only
    # providing the parameters of the classification top to our optimizer)

    loss_func = nn.CrossEntropyLoss()
    opt = optim.Adam(model_ft.parameters(), lr=config.LR)

    # for p in opt.param_groups[0]['params']:
    #     if p.requires_grad:
    #         print(p.shape)

    # calculate steps per epoch for training and validation set
    train_steps = len(train_ds) // config.FEATURE_EXTRACTION_BATCH_SIZE
    val_steps = len(val_ds) // config.FEATURE_EXTRACTION_BATCH_SIZE


    # initialize a dictionary to store training history
    H = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for e in tqdm(range(config.EPOCHS)):
        # set the model in training mode        
        model_ft.train()
        # initialize the total training and validation loss
        total_train_loss = 0.0
        total_val_loss = 0.0
        # initialize the number of correct predictions
        train_correct = 0
        val_correct = 0
        for i, (data, target) in enumerate(train_loader):
            # send the input to the device, batch = [data, target]     
            # torch.Size([4, 3, 224, 224]) torch.Size([4])            
            imgs, lbs = data.to(config.DEVICE), target.to(config.DEVICE)
            # Clear gradients
            opt.zero_grad()
            # perform a forward pass and calculate the training loss            
            preds = model_ft(imgs)
            loss = loss_func(preds, lbs)
            # loss.requires_grad = True

            # backpropagation of gradients          
            loss.backward()
            # Update the parameters
            opt.step()

            # add the loss to the total training loss and
            # calculate the number of correct predictions
            # argmax: returns the indices of the maximum values along an axis.
            
            # print(preds.argmax(axis = 1), lbs)

            total_train_loss += loss.item()
            train_correct += (preds.argmax(axis = 1) == lbs).type(torch.float).sum().item() 
        
        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            model_ft.eval()
            # loop over the validation set
            for (data, target) in val_loader:
                imgs, lbs = data.to(config.DEVICE), target.to(config.DEVICE)
                # make the predictions and calculate the validation loss
                preds = model_ft(imgs)
                total_val_loss += loss_func(preds, lbs)
                # calculate the number of correct predictions
                val_correct += (preds.argmax(1) == lbs).type( torch.float).sum().item()


        # calculate the average training and validation loss
        avg_train_loss = total_train_loss / train_steps
        avg_val_loss = total_val_loss / val_steps
        # calculate the training and validation accuracy
        train_acc = train_correct / len(train_ds)
        val_acc = val_correct / len(val_ds)

        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, config.EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.2f}".format(avg_train_loss, train_acc))
        print("Val loss: {:.6f}, Val accuracy: {:.2f}".format(avg_val_loss, val_acc))

     
        cpu_avg_train_loss = torch.tensor(avg_train_loss, dtype=torch.float32)
        cpu_avg_val_loss = torch.tensor(avg_val_loss, dtype=torch.float32)
        # update our training history
        H["train_loss"].append(cpu_avg_train_loss.clone().detach().cpu().numpy())
        H["train_acc"].append(train_acc)
        H["val_loss"].append(cpu_avg_val_loss.clone().detach().cpu().numpy())
        H["val_acc"].append(val_acc)

    
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.plot(H["train_acc"], label="train_acc")
    plt.plot(H["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(config.WARMUP_PLOT)

    # serialize the model to disk
    # torch.save(model_ft, config.WARMUP_MODEL)

if __name__ == '__main__':
    main()
    