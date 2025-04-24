# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import CONFIG
from model import build_model
from dataset import DRDataset
from utils import set_seed, save_checkpoint

def train():
    set_seed()
    device = torch.device(CONFIG["device"])
    
    model = build_model(CONFIG["model_name"], CONFIG["num_classes"]).to(device)
    train_dataset = DRDataset(CONFIG["train_csv"], CONFIG["image_root"])
    val_dataset = DRDataset(CONFIG["val_csv"], CONFIG["image_root"])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}], Loss: {running_loss/len(train_loader):.4f}")
        save_checkpoint(model, optimizer, epoch, f"{CONFIG['checkpoint_dir']}/model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()