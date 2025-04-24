# dataset.py

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DRDataset(Dataset):
    def __init__(self, csv_file, image_root, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_root = image_root
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = f"{self.image_root}/{self.data.iloc[idx]['image']}"
        image = Image.open(img_path).convert("RGB")
        label = int(self.data.iloc[idx]['level'])

        if self.transform:
            image = self.transform(image)

        return image, label