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

# same transforms (except data mixup) as in Lian et al., "Let it shine: Autofluorescence
# of Papanicolaou-stain improves AI-based cytological oral cancer detection"
bf_transform = T.Compose([
    # pick exactly one of the three with the given probabilities
    T.RandomChoice(
        transforms=[
            T.RandomPosterize(bits=3),
            T.GaussianBlur(kernel_size=5, sigma=1.5),
            T.RandomSolarize(threshold=100),
        ],
        p=[0.4, 0.2, 0.4]
    ),
    T.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.2),
    T.ToTensor(),
])
fl_transform = T.Compose([
    T.ColorJitter(brightness=0.8, contrast=0.8),
    T.GaussianBlur(kernel_size=5, sigma=(0.3, 3.2)),
    T.ToTensor(),
])
joint_transform = T.Compose([
    T.Resize((224,224)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.Normalize(mean=bf_mean+fl_mean, std=bf_std+fl_std),
])

# create datasets for training and validation
train_dataset = EarlyFusionDataset(
    df, split='train', bf_root=bf_root, fl_root=fl_root,
    transforms_bf=bf_transform, transforms_fl=fl_transform, transforms_joint=joint_transform
)
# only resize and normalize val data
val_dataset = EarlyFusionDataset(
    df, split='val', bf_root=bf_root, fl_root=fl_root,
    transforms_bf=T.Compose([T.ToTensor()]), transforms_fl=T.Compose([T.ToTensor()]), 
    transforms_joint=T.Compose([T.Resize(224), T.Normalize(mean=bf_mean+fl_mean, std=bf_std+fl_std)])
)
