import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
import cv2
import albumentations as A

import time
import os
from tqdm.notebook import tqdm


import os
from glob import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torchvision import transforms
import cv2
import albumentations as album

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import numpy as np
from PIL import Image


class AttentionBlock(nn.Module):
    '''
    AttentionBlock Class
    Values:
    channels: number of channels in input
    '''
    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        self.theta = nn.utils.spectral_norm(nn.Conv2d(channels, channels // 8, kernel_size=1, padding=0, bias=False))
        self.phi = nn.utils.spectral_norm(nn.Conv2d(channels, channels // 8, kernel_size=1, padding=0, bias=False))
        self.g = nn.utils.spectral_norm(nn.Conv2d(channels, channels // 2, kernel_size=1, padding=0, bias=False))
        self.o = nn.utils.spectral_norm(nn.Conv2d(channels // 2, channels, kernel_size=1, padding=0, bias=False))

        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x):
        spatial_size = x.shape[2] * x.shape[3]

        # Apply convolutions to get query (theta), key (phi), and value (g) transforms
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), kernel_size=2)
        g = F.max_pool2d(self.g(x), kernel_size=2)

        # Reshape spatial size for self-attention
        theta = theta.view(-1, self.channels // 8, spatial_size)
        phi = phi.view(-1, self.channels // 8, spatial_size // 4)
        g = g.view(-1, self.channels // 2, spatial_size // 4)

        # Compute dot product attention with query (theta) and key (phi) matrices
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), dim=-1)

        # Compute scaled dot product attention with value (g) and attention (beta) matrices
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, self.channels // 2, x.shape[2], x.shape[3]))

        # Apply gain and residual
        return self.gamma * o + x




### The model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down_convolution_1 = ResidualBlock(in_channels, 64)
        self.att0 = AttentionBlock(64)

        self.down_convolution_2 = ResidualBlock(64, 128)
        self.att1 = AttentionBlock(128)

        self.down_convolution_3 = ResidualBlock(128, 256)
        self.att2 = AttentionBlock(256)

        self.down_convolution_4 = ResidualBlock(256, 512)
        self.att3 = AttentionBlock(512)
        
        self.down_convolution_5 = ResidualBlock(512, 1024)
        self.att4 = AttentionBlock(1024)

        self.up_transpose_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_convolution_1 = ResidualBlock(1024, 512)

        self.up_transpose_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_convolution_2 = ResidualBlock(512, 256)

        self.up_transpose_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_convolution_3 = ResidualBlock(256, 128)

        self.up_transpose_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_convolution_4 = ResidualBlock(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        down_1 = self.down_convolution_1(x)
        att0 = self.att0(down_1)
        down_2 = self.max_pool2d(att0)
        
        down_3 = self.down_convolution_2(down_2)
        att1 = self.att1(down_3)
        down_4 = self.max_pool2d(att1)
        
        down_5 = self.down_convolution_3(down_4)
        att2 = self.att2(down_5)
        down_6 = self.max_pool2d(att2)
        
        down_7 = self.down_convolution_4(down_6)
        att3 = self.att3(down_7)
        down_8 = self.max_pool2d(att3)
        
        down_9 = self.down_convolution_5(down_8)
        att4 = self.att4(down_9)

        up_1 = self.up_transpose_1(att4)
        up_2 = self.up_convolution_1(torch.cat([att3, up_1], 1))
        
        up_3 = self.up_transpose_2(up_2)
        up_4 = self.up_convolution_2(torch.cat([att2, up_3], 1))
        
        up_5 = self.up_transpose_3(up_4)
        up_6 = self.up_convolution_3(torch.cat([att1, up_5], 1))
        
        up_7 = self.up_transpose_4(up_6)
        up_8 = self.up_convolution_4(torch.cat([att0, up_7], 1))

        out = self.out(up_8)

        return out
model = UNet().to(device)



#####"### Dataset
def get_training_augmentation():
    train_transform = [
        album.HorizontalFlip(p=0.5),
        album.VerticalFlip(p=0.5),
        album.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
#         album.RandomCrop(height=320, width=320, always_apply=True),
        album.OneOf(
            [
                album.CLAHE(p=1),
                album.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        album.OneOf(
            [
                album.Blur(blur_limit=3, p=1),
                album.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        album.OneOf(
            [
                album.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return album.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        album.PadIfNeeded(384, 480)
    ]
    return album.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
    return album.Compose(_transform)







class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_img=None, transform_msk=None, augmentations=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform_img = transform_img
        self.transform_msk = transform_msk
        self.augmentations = augmentations

        # Define the transformations to be applied to the images and masks
        self.transform_img = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images to [-1, 1]
        ])

        self.transform_msk = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self.image_paths = sorted(glob(os.path.join(self.image_dir, '*.*')))
        self.mask_paths = sorted(glob(os.path.join(self.mask_dir, '*.*')))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        #print(f"Loading image: {img_path}")
        #print(f"Loading mask: {mask_path}")

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            raise FileNotFoundError(f"Image or mask file not found at index {idx}")

        # Load image and convert to grayscale
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image at {img_path}")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Load and transform mask using PIL
        try:
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask)
        except Exception as e:
            raise ValueError(f"Failed to load mask at {mask_path}: {e}")

        # Apply augmentations
        if self.augmentations:
            augmented = self.augmentations(image=gray_image, mask=mask)
            gray_image = augmented['image']
            mask = augmented['mask']

        # Convert the grayscale image to a PIL image and apply transformations
        pil_image = Image.fromarray(gray_image).convert("L")
        image = self.transform_img(pil_image)

        pil_mask = Image.fromarray(mask).convert("L")
        mask = self.transform_msk(pil_mask)

        return image, mask

# Define the paths to the training and testing data
train_image_dir = "./DRIVE/training/images/"
train_mask_dir = "./DRIVE/training/1st_manual/"
test_image_dir = "./DRIVE/test/images/"
test_mask_dir = "./DRIVE/test/1st_manual/"

# Create the dataset with augmentations
train_augmentations = get_training_augmentation()
val_augmentations = get_validation_augmentation()

full_dataset = CustomDataset(train_image_dir, train_mask_dir, augmentations=train_augmentations)

# Define the percentage for splitting (e.g., 80% training, 20% validation)
train_percentage = 0.8
val_percentage = 1 - train_percentage

# Calculate lengths for each split
train_size = int(train_percentage * len(full_dataset))
val_size = len(full_dataset) - train_size

# Split the dataset into training and validation sets
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create DataLoaders for both training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Define the device to be used for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_dataset = CustomDataset(test_image_dir, test_mask_dir)

#Create DataLoaders for both training and validation datasets
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

####" Train
# Setup loss function and optimizer
criterion = nn.BCEWithLogitsLoss()# DiceLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001 )

import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice



import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# Define your loss functions, optimizer, and other components
loss_fn_1 = DiceLoss()
loss_fn_2 = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
torch.manual_seed(42)
epochs = 4000
train_total_losses = []
val_total_losses = []

# Initialize the best validation loss to a high value
best_val_loss = float('inf')

def load_image(filename):
    pixels = np.array(Image.open(filename))
    pixels = ((pixels - 127.5) / 127.5) + 0.5
    return pixels

def display(display_list):
    plt.figure(figsize=(10, 10))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i, image in enumerate(display_list):
        if image is None:
            continue
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        cmap = None
        if i > 0:
            cmap = plt.cm.gray
        plt.imshow(image, cmap=cmap)
        plt.axis('off')
    plt.show()

for epoch in tqdm(range(epochs)):
    train_losses, test_losses = [], []
    print(f"Epoch: {epoch+1} of {epochs}")
    train_loss_1, train_loss_2, train_loss = 0, 0, 0
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss_1 = loss_fn_1(y_pred, y)
        loss_2 = loss_fn_2(y_pred, y)
        loss = loss_1 + loss_2
        train_loss += loss.item()  # accumulate the loss per epoch
        train_loss_1 += loss_1.item()
        train_loss_2 += loss_2.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    train_loss /= len(train_loader)
    train_loss_1 /= len(train_loader)
    train_loss_2 /= len(train_loader)
    test_loss_1, test_loss_2, test_loss = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss_1 = loss_fn_1(y_pred, y)
            loss_2 = loss_fn_2(y_pred, y)
            loss = loss_1 + loss_2
            test_loss += loss.item()
            test_loss_1 += loss_1.item()
            test_loss_2 += loss_2.item()
            test_losses.append(loss.item())
    test_loss /= len(val_loader)
    test_loss_1 /= len(val_loader)
    test_loss_2 /= len(val_loader)
    train_loss = np.average(train_losses)
    train_total_losses.append(train_loss)
    val_loss = np.average(test_losses)
    val_total_losses.append(test_loss)                                                 
    if epoch % 5 == 0:
        print(f"Train loss: {train_loss:.5f}, Dice: {train_loss_1:.5f}, BCE: {train_loss_2:.5f} | Test loss: {test_loss:.5f}, Dice: {test_loss_1:.5f}, BCE: {test_loss_2:.5f}\n")
        # Visualization
        batch = next(iter(val_loader))
        with torch.no_grad():
            model.eval()
            logits = model(batch[0].to(device))
        pr_masks = (torch.sigmoid(logits).squeeze(1) > 0.5).float()
        for image, gt_mask, pr_mask in zip(batch[0], batch[1], pr_masks):
            plt.figure(figsize=(15, 5))
            # Plot Image
            plt.subplot(1, 3, 1)
            plt.imshow(image.permute(1, 2, 0).cpu().numpy(), cmap='gray')
            plt.title("Input Image")
            plt.axis("off")
            # Plot Ground Truth Mask
            plt.subplot(1, 3, 2)
            gt_mask_np = gt_mask.squeeze().cpu().numpy()
            plt.imshow(gt_mask_np, cmap='gray')
            plt.title("Ground Truth Mask")
            plt.axis("off")
            # Plot Prediction Mask
            plt.subplot(1, 3, 3)
            pr_mask_np = pr_mask.squeeze().cpu().numpy()
            plt.imshow(pr_mask_np, cmap='gray')
            plt.title("Predicted Mask")
            plt.axis("off")
            plt.show()
    if epoch % 500 == 0 and epoch != 0:
        torch.save(model.state_dict(), f"./model-{epoch}.pth")
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_total_losses, label='train_loss')
        plt.title("Training & Validation Losses")
        plt.ylabel("Losses")
        plt.xlabel("Epochs")
        plt.legend()

    # Save the model if the validation loss is the best we've seen so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Saved best model with validation loss: {val_loss:.4f}")



#### Load data
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# Define your loss functions, optimizer, and other components
loss_fn_1 = DiceLoss()
loss_fn_2 = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
torch.manual_seed(42)
epochs = 4000
train_total_losses = []
val_total_losses = []

# Initialize the best validation loss to a high value
best_val_loss = float('inf')

def load_image(filename):
    pixels = np.array(Image.open(filename))
    pixels = ((pixels - 127.5) / 127.5) + 0.5
    return pixels

def display(display_list):
    plt.figure(figsize=(10, 10))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i, image in enumerate(display_list):
        if image is None:
            continue
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        cmap = None
        if i > 0:
            cmap = plt.cm.gray
        plt.imshow(image, cmap=cmap)
        plt.axis('off')
    plt.show()

for epoch in tqdm(range(epochs)):
    train_losses, test_losses = [], []
    print(f"Epoch: {epoch+1} of {epochs}")
    train_loss_1, train_loss_2, train_loss = 0, 0, 0
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss_1 = loss_fn_1(y_pred, y)
        loss_2 = loss_fn_2(y_pred, y)
        loss = loss_1 + loss_2
        train_loss += loss.item()  # accumulate the loss per epoch
        train_loss_1 += loss_1.item()
        train_loss_2 += loss_2.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    train_loss /= len(train_loader)
    train_loss_1 /= len(train_loader)
    train_loss_2 /= len(train_loader)
    test_loss_1, test_loss_2, test_loss = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss_1 = loss_fn_1(y_pred, y)
            loss_2 = loss_fn_2(y_pred, y)
            loss = loss_1 + loss_2
            test_loss += loss.item()
            test_loss_1 += loss_1.item()
            test_loss_2 += loss_2.item()
            test_losses.append(loss.item())
    test_loss /= len(val_loader)
    test_loss_1 /= len(val_loader)
    test_loss_2 /= len(val_loader)
    train_loss = np.average(train_losses)
    train_total_losses.append(train_loss)
    val_loss = np.average(test_losses)
    val_total_losses.append(test_loss)                                                 
    if epoch % 5 == 0:
        print(f"Train loss: {train_loss:.5f}, Dice: {train_loss_1:.5f}, BCE: {train_loss_2:.5f} | Test loss: {test_loss:.5f}, Dice: {test_loss_1:.5f}, BCE: {test_loss_2:.5f}\n")
        # Visualization
        batch = next(iter(val_loader))
        with torch.no_grad():
            model.eval()
            logits = model(batch[0].to(device))
        pr_masks = (torch.sigmoid(logits).squeeze(1) > 0.5).float()
        for image, gt_mask, pr_mask in zip(batch[0], batch[1], pr_masks):
            plt.figure(figsize=(15, 5))
            # Plot Image
            plt.subplot(1, 3, 1)
            plt.imshow(image.permute(1, 2, 0).cpu().numpy(), cmap='gray')
            plt.title("Input Image")
            plt.axis("off")
            # Plot Ground Truth Mask
            plt.subplot(1, 3, 2)
            gt_mask_np = gt_mask.squeeze().cpu().numpy()
            plt.imshow(gt_mask_np, cmap='gray')
            plt.title("Ground Truth Mask")
            plt.axis("off")
            # Plot Prediction Mask
            plt.subplot(1, 3, 3)
            pr_mask_np = pr_mask.squeeze().cpu().numpy()
            plt.imshow(pr_mask_np, cmap='gray')
            plt.title("Predicted Mask")
            plt.axis("off")
            plt.show()
    if epoch % 500 == 0 and epoch != 0:
        torch.save(model.state_dict(), f"./model-{epoch}.pth")
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_total_losses, label='train_loss')
        plt.title("Training & Validation Losses")
        plt.ylabel("Losses")
        plt.xlabel("Epochs")
        plt.legend()

    # Save the model if the validation loss is the best we've seen so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Saved best model with validation loss: {val_loss:.4f}")




### Inference

import torch
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'val_loader' is your DataLoader
batch = next(iter(val_loader))

with torch.no_grad():
    model.eval()
    logits = model(batch[0].to(device))  # Assuming image is at index 0
pr_masks = (torch.sigmoid(logits).squeeze(1) > 0.5).float()  # Binary predictions

# Iterate through the batches
for image, gt_mask, pr_mask in zip(batch[0], batch[1], pr_masks):
    plt.figure(figsize=(15, 5))

    # Plot Image
    plt.subplot(1, 3, 1)
    plt.imshow(image.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.title("Image")
    plt.axis("off")

    # Plot Ground Truth Mask
    plt.subplot(1, 3, 2)
    gt_mask_np = gt_mask.squeeze().cpu().numpy()
    plt.imshow(gt_mask_np, cmap='gray')
    plt.title("Ground Truth")
    plt.axis("off")

    # Plot Prediction Mask
    plt.subplot(1, 3, 3)
    pr_mask_np = pr_mask.cpu().numpy()
    plt.imshow(pr_mask_np, cmap='gray')
    plt.title("Prediction")
    plt.axis("off")

    plt.show()

