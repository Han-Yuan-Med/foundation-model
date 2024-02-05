import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage import io
import matplotlib.image as img
import numpy as np
from sklearn import metrics
import cv2
import torchxrayvision as xrv
import skimage
import random
import torchvision.transforms.functional as trans
from PIL import Image


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Lung_cls(Dataset):
    def __init__(self, csv_file, img_dir):
        self.annotations = csv_file
        self.img_dir = img_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # grab the image path from the current index
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))
        label = torch.tensor(int(self.annotations.iloc[index, 4]))

        image_fused = np.zeros((3, 224, 224))
        image_fused[0] = image / 255
        image_fused[1] = image / 255
        image_fused[2] = image / 255
        image_fused = torch.from_numpy(image_fused)

        return image_fused, label


class Lung_cls_remedis(Dataset):
    def __init__(self, csv_file, img_dir):
        self.annotations = csv_file
        self.img_dir = img_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # grab the image path from the current index
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (448, 448))
        label = torch.tensor(int(self.annotations.iloc[index, 4]))

        image_fused = np.zeros((448, 448, 3))
        image_fused[:, :, 0] = image / 255
        image_fused[:, :, 1] = image / 255
        image_fused[:, :, 2] = image / 255

        return image_fused, label


class Lung_cls_xrv(Dataset):
    def __init__(self, csv_file, img_dir):
        self.annotations = csv_file
        self.img_dir = img_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # grab the image path from the current index
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = skimage.io.imread(img_path)
        image = xrv.datasets.normalize(image, 255)
        image = cv2.resize(image, (224, 224))
        label = torch.tensor(int(self.annotations.iloc[index, 4]))

        image = torch.from_numpy(image)

        return image, label


class Lung_cls_imagenet(Dataset):
    def __init__(self, csv_file, img_dir):
        self.annotations = csv_file
        self.img_dir = img_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # grab the image path from the current index
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))
        label = torch.tensor(int(self.annotations.iloc[index, 4]))
        image_fused = np.zeros((3, 224, 224))
        image_fused[0, :, :] = image / 255
        image_fused[1, :, :] = image / 255
        image_fused[2, :, :] = image / 255
        image = trans.normalize(torch.from_numpy(image_fused), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image, label


def train_cls(cls_model, train_loader, criterion, optimizer, device):
    cls_model.train()
    for batch_idx, (sample, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        sample = sample.to(device=device).float()
        targets = targets.to(device=device)

        # forward
        scores = cls_model(sample)
        loss = criterion(scores, targets)
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()


def valid_cls(cls_model, val_loader, device):
    prob_list = []
    label_list = []
    with torch.no_grad():
        for data in val_loader:
            images, labels = data[0].float().to(device), data[1].to(device)
            outputs = cls_model(images)
            label_list = np.concatenate((label_list, labels.cpu().numpy()), axis=None)
            prob_list = np.concatenate((prob_list, torch.sigmoid(outputs)[:, 1].detach().cpu().numpy()), axis=None)
    fpr, tpr, thresholds = metrics.roc_curve(label_list, prob_list)
    return metrics.auc(fpr, tpr)


def valid_cls_prc(cls_model, val_loader, device):
    prob_list = []
    label_list = []
    with torch.no_grad():
        for data in val_loader:
            images, labels = data[0].float().to(device), data[1].to(device)
            outputs = cls_model(images)
            label_list = np.concatenate((label_list, labels.cpu().numpy()), axis=None)
            prob_list = np.concatenate((prob_list, torch.sigmoid(outputs)[:, 1].detach().cpu().numpy()), axis=None)
    precision, recall, thresholds = metrics.precision_recall_curve(label_list, prob_list)
    return metrics.auc(recall, precision)

