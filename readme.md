# This repo includes 2 examples for the image classification using CNN networks.

This project was written using torch, opencv libraries. Please check these requirements before the start.
This project structure with several directories and py files, especially.
- The chip folder contains some files for the configurations and common methods.
- The images folder contains a number of sample images where we will apply the CNN network to clasify.

1. Image Classification with a pre-trained model.

- use a pretrained VGG16 model
- get the top-5 result,
- the the top-1 result,
- load and preprocess the image with opencv, numpy,
- using GPU computing.

2. Transfer learning with PyTorch.

Transfer learning includes: Feature extraction and Fine-tuning

The describle of the libraries that we used in this project was introduced as below.
- tqdm for making a process bar during the training
- imutils for processing basic image processing functions such as translation, rotation, resizing, skeletonization
- paths: A submodule of imutils used to gather paths to images inside a given directory
- shutil to copy files from one location to another

2.1. Transfer learning via feature extraction works by:

- Taking a pre-trained CNN (e.g., VGG16)
- Removing the FC layer head from the CNN
- Treating the output of the body of the network as an arbitrary feature extractor with spatial dimensions M × N × C

After that, we can have 2 options:
- Take a standard Logistic Regression classifier (like the one found in the scikit-learn library) and train it on the extracted features from each image
- Or, more simply, place a softmax classifier on top of the body of the network. It is a good choice.

2.2. Transfer learning with Fine-tuning statergy:

Similar to feature extraction, we start by removing the FC layer head from the network, but this time we create a brand new layer head with a set of linear, ReLU, and dropout layers, similar to what you would see on a modern state-of-the-art CNN.

