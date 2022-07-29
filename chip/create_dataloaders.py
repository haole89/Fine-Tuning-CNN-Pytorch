from . import config
from torch.utils.data import DataLoader
from torchvision import datasets
import os

# Note: shuffle data for training but not for validation.
# num_workers: 1 or 2 is enough
def get_dataloader(root_path, transforms, batch_size, shuffle = True):
    # create a dataset and use it to create a data loader
    ds = datasets.ImageFolder(
        root=root_path,
        transform = transforms
    )

    loader = DataLoader(
        dataset=ds,
        batch_size = batch_size,
        shuffle=shuffle,
        num_workers= 1
    )
    # return a tuple of  the dataset and the data loader
    return (ds, loader)




