import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch

class EarlyFusionDataset(Dataset):
    def __init__(self, df, split, bf_root='/content/data/BF/train', fl_root='/content/data/FL/train', transforms_bf=None, transforms_fl=None):
        """
        Create a dataset where each sample consists of the pair of BF and FL images stacked in the channel dimension.

        df        : DataFrame with columns ['Name','Diagnosis','patient_id','split']
        bf_root   : path to BF images folder (train or test)
        fl_root   : path to FL images folder (train or test)
        split     : 'train' or 'val'
        transforms_bf : transforms for BF images (optional)
        transforms_fl : transforms for FL images (optional)
        """
        self.subset = df[df['split'] == split].reset_index(drop=True)
        self.bf_root = bf_root
        self.fl_root = fl_root
        self.transforms_bf = transforms_bf
        self.transforms_fl = transforms_fl

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        row = self.subset.iloc[idx]
        name = row['Name']
        label = torch.tensor(row['Diagnosis'], dtype=torch.float32)

        # Load images
        bf_path = os.path.join(self.bf_root, name)
        fl_path = os.path.join(self.fl_root, name)
        bf_img = Image.open(bf_path).convert('RGB')
        fl_img = Image.open(fl_path).convert('RGB')

        # Apply transforms
        if self.transforms_bf:
            bf_img = self.transforms_bf(bf_img)
        else:
            bf_img = T.ToTensor()(bf_img)
        if self.transforms_fl:
            fl_img = self.transforms_fl(fl_img)
        else:
            fl_img = T.ToTensor()(fl_img)


        # Stack images along the channel dimension
        x = torch.cat([bf_img, fl_img], dim=0)

        return x, label



