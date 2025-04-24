# train.py
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from config import *
from model import HybridModel
from dataset import RetinaDataset
from utils import get_transforms



def train_model():
    train_tf, val_tf = get_transforms()
    train_ds = RetinaDataset(TRAIN_PATH, train_tf)
    val_ds   = RetinaDataset(VALID_PATH, val_tf)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=4)

    model = HybridModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_kappa, early_stop = -1, 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds==labels).sum().item()
            total += labels.size(0)
      
    return model

if __name__ == "__main__":
    train_model()
