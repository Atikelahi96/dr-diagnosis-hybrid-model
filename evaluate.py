# evaluate.py

import torch
from torch.utils.data import DataLoader
from config import CONFIG
from model import build_model
from dataset import DRDataset

def evaluate():
    device = torch.device(CONFIG["device"])
    model = build_model(CONFIG["model_name"], CONFIG["num_classes"]).to(device)
    model.load_state_dict(torch.load(f"{CONFIG['checkpoint_dir']}/model_epoch_25.pth")["model_state_dict"])
    model.eval()

    val_dataset = DRDataset(CONFIG["val_csv"], CONFIG["image_root"])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Accuracy: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    evaluate()