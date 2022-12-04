import numpy as np
import pandas as pd

from pathlib import Path

import torch
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image

class CustomDataset(Dataset):
    """Custom dataset"""

    def __init__(self, dataset_path, csv_file, transform=None, target_transform=None):
        
        self.sample_dim = pd.read_csv(csv_file, header=None, nrows=1).values.tolist()[0] # First element from the list obtained from the conversion of a the Dataframe of metadata csv[:1]
        self.img_labels = pd.read_csv(csv_file, header=None, skiprows=1)

        self.dataset_path = dataset_path

        self.transform = transform
        self.target_transform = target_transform

    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        img_path = self.dataset_path/img_name
        img = read_image(str(img_path))

        label = np.array([self.img_labels.iloc[idx, 1:]], dtype='float')
        #print(label)
        label = self.reduction(label)

        if self.transform:
            img = transform(img)

        if self.target_transform:
            label = target_transform(label)
        
        return {'image': img, 'label': label}
    
    def reduction(self, label):
        reduced_label = np.zeros_like(label)
        nbr_dim = len(self.sample_dim)

        #TODO: handle len(label)%nbr_dim = len(label)%nbr_dim case != 0

        for i in range(label.shape[-1]//nbr_dim):
            point = label[:, i*nbr_dim : (i+1)*nbr_dim]
            reduced_point = point/self.sample_dim
            #print(point, self.sample_dim, reduced_point)
            reduced_label[:, i*nbr_dim : (i+1)*nbr_dim] = reduced_point
        
        return reduced_label