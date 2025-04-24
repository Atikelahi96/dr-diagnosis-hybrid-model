# evaluate.py

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from config import *
from model import HybridModel
from dataset import RetinaDataset
from utils import get_transforms

if __name__ == "__main__":
    model = HybridModel().to(DEVICE)
    model.load_state_dict(torch.load('best_hybrid.pth'))
    model.eval()
    _, test_tf = get_transforms()
    test_ds = RetinaDataset(TEST_PATH, test_tf)
    test_loader = DataLoader(test_ds, BATCH_SIZE)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            outs = model(imgs.to(DEVICE))
            preds = outs.argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    print(classification_report(all_labels, all_preds))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.savefig('confusion.png')
