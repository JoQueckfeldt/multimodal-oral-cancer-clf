import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch

class EarlyFusionDataset(Dataset):
    def __init__(self, df, split, bf_root='/content/data/BF/train', fl_root='/content/data/FL/train', transforms_bf=None, transforms_fl=None, transforms_joint=None):
        """
        Create a dataset where each sample consists of the pair of BF and FL images stacked in the channel dimension.

        df : DataFrame with columns ['Name','Diagnosis','patient_id','split']
        bf_root : path to BF images folder (train or test)
        fl_root : path to FL images folder (train or test)
        split : 'train' or 'val'
        transforms_bf : BF-specific transforms
        transforms_fl : FL-specific transforms
        transforms_joint : joint transforms for both modalities (e.g., geometric transformations, normalization)
        """
        self.subset = df[df['split'] == split].reset_index(drop=True)
        self.bf_root = bf_root
        self.fl_root = fl_root
        self.transforms_bf = transforms_bf
        self.transforms_fl = transforms_fl
        self.transforms_joint = transforms_joint

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

        # Apply modality-specific transforms 
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

        # Apply joint transforms (e.g. geometric transformations and normalization)
        if self.transforms_joint:
            x = self.transforms_joint(x)
        
        return x, label

class BF3ChannelDataset(Dataset):
    def __init__(self, df, bf_root, transform=None):
        """
        df: DataFrame with columns ['Name','Diagnosis','patient_id','split']
        bf_root: directory containing BF RGB images (e.g. 3 channels)
        transform: torchvision transforms to apply (e.g. augment+normalize)
        """
        self.df = df.reset_index(drop=True)
        self.bf_root = bf_root
        self.transform = transform
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row['Name']
        label = torch.tensor(row['Diagnosis'], dtype=torch.float32)

        img_path = os.path.join(self.bf_root, name)
        img = Image.open(img_path).convert('RGB')  # BF stored as RGB

        if self.transform:
            img = self.transform(img)

        return img, label



