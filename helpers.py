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

    # filter for training set
    df_train = df[df['split']=='train']['Name']

    # accumulators for BF and FL
    bf_sum = torch.zeros(3)
    bf_sum_sq = torch.zeros(3)
    fl_sum = torch.zeros(3)
    fl_sum_sq = torch.zeros(3)
    n_pixels = 0

    for name in df_train:
        # convert images to tensors (C, H, W)
        bf = T.ToTensor()(Image.open(os.path.join(bf_root, name)).convert('RGB'))
        fl = T.ToTensor()(Image.open(os.path.join(fl_root, name)).convert('RGB'))

        C, H, W = bf.shape
        pixels = H * W
        n_pixels += pixels

        # sum over H and W → shape (C,)
        bf_sum += bf.sum(dim=[1,2])
        bf_sum_sq += (bf * bf).sum(dim=[1,2])
        fl_sum += fl.sum(dim=[1,2])
        fl_sum_sq += (fl * fl).sum(dim=[1,2])

    # compute mean and std
    bf_mean = bf_sum / n_pixels
    bf_var = bf_sum_sq / n_pixels - bf_mean**2
    bf_std = torch.sqrt(bf_var)

    fl_mean = fl_sum / n_pixels
    fl_var = fl_sum_sq / n_pixels - fl_mean**2
    fl_std = torch.sqrt(fl_var)

    return bf_mean.tolist(), bf_std.tolist(), fl_mean.tolist(), fl_std.tolist()