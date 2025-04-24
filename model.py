# model.py

import torch.nn as nn
import torchvision.models as models

def build_model(model_name="resnet50", num_classes=5, pretrained=True):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    return model