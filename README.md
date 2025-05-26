# multimodal-oral-cancer-clf

## Example: create early‐fusion dataset

```python
import torch
import torchvision.transforms as T
import pandas as pd
from torch.utils.data import DataLoader

from dataset import EarlyFusionDataset
from helpers import calculate_mean_std

# example if working in colab and have the data stored in 'data' folder
bf_root = '/content/data/BF/train'  # Brightfield images root directory
fl_root = '/content/data/FL/train'  # Fluorescence images root directory
train_csv_path = '/content/data/train.csv'  # Path to the CSV file with patient data

# create DataFrame with columns ['Name','Diagnosis','patient_id','split']
val_patients_id = ['05', '07'] # patient IDs for validation set
df = pd.read_csv(train_csv_path) # colums: [Name, Diagnosis]. Names are like pat_NN_image_X
df['patient_id'] = df['Name'].apply(lambda x: x.split('_')[1])  # Extract patient_id from Name
df['split'] = df['patient_id'].apply(lambda x: 'val' if x in val_patients_id else 'train') # Assign split based on patient_id


# calculate statistics for normalization based on the images with split='train'
bf_mean, bf_std, fl_mean, fl_std = calculate_mean_std(df, bf_root=bf_root, fl_root=fl_root)

# define transforms for BF and FL images (we might want to use different transforms for each type of image)
# Add more transformations, such as data augmentation
bf_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=bf_mean, std=bf_std)
])

fl_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=fl_mean, std=fl_std)
])

# create datasets for training and validation
train_dataset = EarlyFusionDataset(df, split='train', bf_root=bf_root, fl_root=fl_root,
                                   bf_transform=bf_transform, fl_transform=fl_transform)
val_dataset = EarlyFusionDataset(df, split='val', bf_root=bf_root, fl_root=fl_root,
                                  bf_transform=bf_transform, fl_transform=fl_transform)
