# utils.py

import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import RetinaDataset
from config import TRAIN_PATH, VALID_PATH, BATCH_SIZE


def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    mean, std = torch.zeros(3), torch.zeros(3)
    for imgs, _ in loader:
        imgs = imgs.to(torch.float32)/255.0
        imgs = imgs.permute(0,3,1,2) if imgs.ndim==4 else imgs
        mean += imgs.mean(dim=[0,2,3])
        std  += imgs.std(dim=[0,2,3])
    mean /= len(loader)
    std  /= len(loader)
    return mean.numpy(), std.numpy()


def get_transforms():
    train_ds = RetinaDataset(TRAIN_PATH, transform=None)
    mean, std = compute_mean_std(train_ds)
    train_tf = A.Compose([
        A.Resize(512,512), A.RandomCrop(448,448), A.HorizontalFlip(), A.VerticalFlip(),
        A.Rotate(limit=90, p=0.3), A.RandomBrightnessContrast(p=0.3), A.CLAHE(p=0.5),
        A.Normalize(mean.tolist(), std.tolist()), ToTensorV2()
    ])
    val_tf = A.Compose([A.Resize(448,448), A.Normalize(mean.tolist(), std.tolist()), ToTensorV2()])
    return train_tf, val_tf
