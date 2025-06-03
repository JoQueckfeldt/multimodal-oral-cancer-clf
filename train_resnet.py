import torch
import torchvision.transforms as T
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn, optim
from torchvision.models import resnet18, resnet50
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC

from dataset import EarlyFusionDataset, BF3channelDataset
from helpers import calculate_mean_std, mixup_data, mixup_bce_loss

# Mount Google Drive if running on Google Colab
#from google.colab import drive
#drive.mount('/content/drive')

data_path = '/content/drive/MyDrive/multimodal-cancer-classification-challenge-2025'

# for saving best model checkpoint
model_ckpt_path = '/content/drive/MyDrive/resnet18_pretr.pth'

bf_root = '/content/data/BF/train'  # Brightfield images root directory
fl_root = '/content/data/FL/train'  # Fluorescence images root directory
train_csv_path = '/content/data/train.csv'  # Path to the CSV file with patient data

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

##### HYPERPARAMETERS #####
n_epochs = 30
batch_size = 256
dropout=0.3
learning_rate = 8e-5
weight_decay = 0.1
###########################

# change to resnet if needed
model = resnet18(pretrained=True)
model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Sequential(
    nn.Dropout(dropout),
    nn.Linear(model.fc.in_features, 1)
)
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.7,
    patience=3,
    min_lr=1e-7,
)

# create DataFrame with columns ['Name','Diagnosis','patient_id','split']
val_patients_id = ['03', '07', '09'] # patient IDs for validation set
df = pd.read_csv(train_csv_path) # colums: [Name, Diagnosis]. Names are like pat_NN_image_X
df['patient_id'] = df['Name'].apply(lambda x: x.split('_')[1])  # Extract patient_id from Name
df['split'] = df['patient_id'].apply(lambda x: 'val' if x in val_patients_id else 'train') # Assign split based on patient_id

# calculate statistics for normalization based on the images with split='train'
bf_mean, bf_std, fl_mean, fl_std = calculate_mean_std(df, bf_root=bf_root, fl_root=fl_root)

# same transforms as in Lian et al., "Let it shine: Autofluorescence
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

# create datasets for training and validation (change to BF3channelDataset if you want to use only BF images)
train_dataset = EarlyFusionDataset(
    df, split='train', bf_root=bf_root, fl_root=fl_root,
    transforms_bf=bf_transform, transforms_fl=fl_transform, transforms_joint=joint_transform
)
# only resize and normalize val data
val_dataset = EarlyFusionDataset(
    df, split='val', bf_root=bf_root, fl_root=fl_root,
    transforms_bf=T.Compose([T.ToTensor()]), transforms_fl=T.Compose([T.ToTensor()]), 
    transforms_joint=T.Compose([T.Resize((224,224)), T.Normalize(mean=bf_mean+fl_mean, std=bf_std+fl_std)])
)

# data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# weighted BCE loss for imbalanced dataset
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.7]).to(device))

# for comparison with the validation loss
criterion_print = nn.BCEWithLogitsLoss()  # No pos_weight

# metrics
acc_metric = Accuracy(task="binary", threshold=0.5).to(device)
prec_metric = Precision(task="binary", threshold=0.5).to(device)
recall_metric = Recall(task="binary", threshold=0.5).to(device)
f1_metric = F1Score(task="binary", threshold=0.5).to(device)
auc_metric = AUROC(task="binary").to(device)

train_losses = []
train_mixup_losses = []
val_losses = []
accs = []
precs = []
recalls = []
f1s = []
aucs = []

best_loss = float('inf')  

for epoch in range(1, n_epochs + 1):
    model.train()
    running_loss = 0.0
    running_mixup_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        ###
        x_mix, y_a, y_b, lam = mixup_data(x, y, alpha=0.8)
        ###
        optimizer.zero_grad()
        logits = model(x_mix).view(-1)
        loss = mixup_bce_loss(logits, y_a, y_b, lam, criterion)
        
        with torch.no_grad():
            logits_print = model(x).view(-1) # unmixed, unweighted train bce loss for comparison to validation loss
            loss_print   = criterion_print(logits_print, y)

        loss.backward() # update weights based on mixup bce loss
        optimizer.step()
        running_loss += loss_print.item() * x.size(0)
        running_mixup_loss += loss.item() * x.size(0)

    epoch_mixup_loss = running_mixup_loss / len(train_loader.dataset)
    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    train_mixup_losses.append(epoch_mixup_loss)

    # Validation
    model.eval()
    val_running_loss = 0.0
    acc_metric.reset()
    prec_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    auc_metric.reset()

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).view(-1)
            loss = criterion_print(logits, y)
            val_running_loss += loss.item() * x.size(0)

            probs = torch.sigmoid(logits)
            acc_metric.update(probs, y.int())
            prec_metric.update(probs, y.int())
            recall_metric.update(probs, y.int())
            f1_metric.update(probs, y.int())
            auc_metric.update(probs, y.int())

    epoch_val_loss = val_running_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    if epoch_val_loss < best_loss:
        best_loss = epoch_val_loss
        torch.save(model.state_dict(), model_ckpt_path)

    scheduler.step(epoch_val_loss)
  
    # compute and store metrics
    epoch_acc = acc_metric.compute().item()
    epoch_prec = prec_metric.compute().item()
    epoch_rec = recall_metric.compute().item()
    epoch_f1 = f1_metric.compute().item()
    epoch_auc = auc_metric.compute().item()

    accs.append(epoch_acc)
    precs.append(epoch_prec)
    recalls.append(epoch_rec)
    f1s.append(epoch_f1)
    aucs.append(epoch_auc)

    print(f"Epoch {epoch:02d} | Mixup Loss: {epoch_mixup_loss:.4f} | "
          f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
          f"Acc: {epoch_acc:.4f} | Prec: {epoch_prec:.4f} | Recall: {epoch_rec:.4f} | "
          f"F1: {epoch_f1:.4f} | AUC: {epoch_auc:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']}")
    

# plotting ----------------------------
import matplotlib.pyplot as plt

epochs = range(1, n_epochs)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses,   label='Val Loss')
plt.title('Loss')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(epochs, accs,  label='Accuracy')
plt.title('Accuracy')

plt.subplot(2, 3, 3)
plt.plot(epochs, precs, label='Precision')
plt.title('Precision')

plt.subplot(2, 3, 4)
plt.plot(epochs, recalls, label='Recall')
plt.title('Recall')

plt.subplot(2, 3, 5)
plt.plot(epochs, f1s,    label='F1 Score')
plt.title('F1 Score')

plt.subplot(2, 3, 6)
plt.plot(epochs, aucs,   label='ROC AUC')
plt.title('ROC AUC')

plt.tight_layout()
plt.show()