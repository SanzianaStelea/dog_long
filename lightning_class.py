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
from torch.nn import functional as F
import pytorch_lightning as pl
from torch import nn

"""We redefine the model class so that it will use the Pytorch Lightning module. The Dataloader model is also the same as in classes.py
but this time it will use the maximum CPU workers"""

class model_lightning (pl.LightningModule):

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

    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)
  
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer



from torch.utils.data.sampler import SubsetRandomSampler

class CustomDataLoader:

    def __init__(self, batch_size, validation_split, test_split, dataset, random_seed=42):
        
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

        train_sampler = SubsetRandomSampler(self.train_indices)
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size, 
                                           sampler=train_sampler, num_workers=8)
        return train_loader
        
    def validation_loader (self):

        validation_sampler = SubsetRandomSampler(self.val_indices)
        validation_loader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size, 
                                           sampler=validation_sampler, num_workers=8)
        return validation_loader
    
    def test_sampler (self):

        test_sampler = SubsetRandomSampler(self.test_indices)
        test_loader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size, 
                                           sampler=test_sampler, num_workers=8)
        return test_loader
