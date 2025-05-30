import os
import pandas as pd
import numpy as np
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


def mixup_data(x, y, alpha=0.8):
    """
    Returns mixed inputs, and mixed scalar labels for BCE.

    x:      Tensor of shape [B, C, H, W]
    y:      Tensor of shape [B], dtype float (0.0 or 1.0)
    alpha:  mixup hyperparam

    outputs:
      x_mix:   [B, C, H, W]
      y_mix:   [B], values in [0,1]
      lam:     mix coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    B = x.size(0)
    idx = torch.randperm(B, device=x.device)

    x_a, x_b = x, x[idx]
    y_a, y_b = y, y[idx]

    x_mix = lam * x_a + (1.0 - lam) * x_b
    return x_mix, y_a, y_b, lam


def mixup_bce_loss(logits, y_a, y_b, lam, criterion):
    """
    logits:     Tensor [B] of raw model outputs
    y_a, y_b:   Tensors [B] of 0/1 labels for the two shuffled batches
    lam:        scalar mixup coefficient
    criterion:  BCE loss function, nn.BCEWithLogitsLoss(pos_wheight=num_negatives/num_positives)
    
    Returns:
        mixed loss: weighted sum of BCE losses for the two sides
    """
    # sigmoid probabilities
    p = torch.sigmoid(logits)

    # BCE for side a
    loss_a = criterion(logits, y_a)
    # BCE for side b
    loss_b = criterion(logits, y_b)

    # weighted sum
    return lam * loss_a + (1.0 - lam) * loss_b