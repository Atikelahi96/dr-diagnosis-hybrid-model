# dataset.py

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

distclass RetinaDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=(448,448)):
        self.root_dir    = root_dir
        self.transform   = transform
        self.image_size  = image_size
        self.classes     = sorted(os.listdir(root_dir))
        self.samples = []
        for idx, cls in enumerate(self.classes):
            p = os.path.join(root_dir, cls)
            for fname in os.listdir(p):
                self.samples.append((os.path.join(p,fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB").resize(self.image_size)
        img = np.array(img)
        if self.transform:
            img = self.transform(image=img)["image"]
        if isinstance(img, np.ndarray):
            img = torch.tensor(img, dtype=torch.float32).permute(2,0,1)
        return img, label
