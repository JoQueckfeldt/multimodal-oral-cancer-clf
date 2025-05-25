import os
import pandas as pd
from PIL import Image
import torchvision.transforms as T
import torch

def calculate_mean_std(df, bf_root='/content/data/BF/train', fl_root='/content/data/FL/train'):
    """
    Calculate the channel-wise mean and standard deviation fro the training images.

    Args
    -----
        df : DataFrame with columns ['Name','Diagnosis','patient_id','split']
        bf_root : path to BF images folder
        fl_root : path to FL images folder
    
    Returns
    -------
        bf_mean, bf_std, fl_mean, fl_std : lists of channel-wise means and stds
    """
    df_train = df[df['split'] == 'train'].reset_index(drop=True)
    bf_images = []
    fl_images = []

    for name in df_train['Name']:
        bf_path = os.path.join(bf_root, name)
        fl_path = os.path.join(fl_root, name)

        bf_img = Image.open(bf_path).convert('RGB')
        fl_img = Image.open(fl_path).convert('RGB')
        bf_images.append(T.ToTensor()(bf_img))
        fl_images.append(T.ToTensor()(fl_img))
    
    bf_images = torch.stack(bf_images) # shape (N, C, H, W)
    fl_images = torch.stack(fl_images)

    # Calculate mean and std for each channel
    bf_mean = bf_images.mean(dim=[0, 2, 3])
    bf_std = bf_images.std(dim=[0, 2, 3])
    fl_mean = fl_images.mean(dim=[0, 2, 3])
    fl_std = fl_images.std(dim=[0, 2, 3])

    return bf_mean.tolist(), bf_std.tolist(), fl_mean.tolist(), fl_std.tolist()