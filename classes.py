
"""This is a file that contains custom classes and functions that we will use for training networks. Classes in this file 
do not use Pytorch Lightning"""

import os
import cv2
import numpy as np
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from PIL import Image
import pytorch_lightning as pl

class CustomImageDataset(Dataset):
    """
    Class for custom dataset. It is meant to support creating a new dataset using locally stored images that are labeled in a dataframe.
    """
    def __init__(self, annotations_file, img_dir, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.CenterCrop(212),
                                transforms.Resize(64),
                                ]), target_transform=None):
        """
        The inputs : annotation_file is a dataframe and img_dir is a path. 
        annotation_file should have on the first column the image_id (for example 14446.jpg) that represents the name under which the images is saved. On the second column should be the labels.
        img_dir is the path to the directory where the images are saved.
        Function intitialises the object.
        """
        self.img_labels = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        Returns the dimension of the dataset.
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Parameter idx is an integer. Function returns the image stored as tensor and its label, corresponding to the index idx of the dataframe.
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = torch.tensor(self.img_labels.iloc[idx, 1], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



from torch.utils.data.sampler import SubsetRandomSampler

class CustomDataLoader:

    """This is a class for custom train, validation and test dataloaders"""

    def __init__(self, batch_size, validation_split, test_split, dataset, random_seed=42):
        
        """The inputs are: batch_size (integer), validation_split, test_split (numbers between 0 and 1), dataset is a variable 
        that is part of the Dataset class, and random seed an integer number (default 42) used for the deterministic data splitting"""

        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.dataset = dataset
        self.random_seed = random_seed

        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split_val = int(np.floor(self.validation_split * dataset_size))
        split_test= int(np.floor(self.test_split * dataset_size))

        if self.random_seed :
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)

        self.train_indices, self.val_indices, self.test_indices = indices[split_val+split_test:], indices[:split_val], indices[split_val:split_val+split_test]

    def train_loader (self):
        
        """Returns the train loader"""

        train_sampler = SubsetRandomSampler(self.train_indices)
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size, 
                                           sampler=train_sampler)
        return train_loader
        
    def validation_loader (self):
        
        """Returns the validation loader"""

        validation_sampler = SubsetRandomSampler(self.val_indices)
        validation_loader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size, 
                                           sampler=validation_sampler)
        return validation_loader
    
    def test_sampler (self):
        
        """Returns the test loader"""

        test_sampler = SubsetRandomSampler(self.test_indices)
        test_loader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size, 
                                           sampler=test_sampler)
        return test_loader

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim

class model1(pl.LightningModule):

    """Neural network model with 2 CNN and 2 deep layers. Activation function is RELU."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704 , 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def train_model(model, lr, momentum, epochs, train_loader):

    """Functions that trains models with CrossEntropyLoss and SGD optimizer.
    It takes as parameters the neural network model, learning rate and momentum for SGD, number of epochs and the train loader."""

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr, momentum)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 500 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0
                
    print('Finished Training')

def evaluate_model(model, loader):

    """Function that evaluates a model. It takes as parameters the model and the data loader."""
    correct = 0
    total = 0
    model.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    print(f'Accuracy of the network on the validation/test images: {100 * correct // total} %')


def ilustrate_data (database, classes, figure_number):

    n = len(classes)
    class_list = [0]*n
    final_list = [1]*n
    plt.figure(figure_number)

    for i in range(len(database)):
        image, label= database.__getitem__(i)

        for j in range (len (class_list) ):
            if label==j and class_list[j]==0:
                plt.subplot(1, n, j)       
                plt.imshow(  image.permute(1, 2, 0), cmap='gray'  )
                plt.title ("{j}-{classes[j]}" )
                class_list [j] = 1
        
        if(class_list == final_list):
            break
