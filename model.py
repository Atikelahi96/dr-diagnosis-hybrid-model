# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1)
        self.bn1   = nn.BatchNorm2d(mip)
        self.act   = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0,1,3,2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0,1,3,2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_h * a_w

class HybridModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.effnet = torchvision.models.efficientnet_b3(pretrained=True)
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.effnet.classifier = nn.Identity()
        self.resnet.fc         = nn.Identity()
        self.ca_eff = CoordinateAttention(1536)
        self.ca_res = CoordinateAttention(2048)
        self.classifier = nn.Sequential(
            nn.Linear(1536+2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        eff = self.effnet.features(x)
        eff = self.ca_eff(eff)
        eff = F.adaptive_avg_pool2d(eff, (1,1)).flatten(1)
        res = self.resnet.conv1(x)
        res = self.resnet.bn1(res)
        res = self.resnet.relu(res)
        res = self.resnet.maxpool(res)
        for layer in [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]:
            res = layer(res)
        res = self.ca_res(res)
        res = F.adaptive_avg_pool2d(res, (1,1)).flatten(1)
        combined = torch.cat([eff, res], dim=1)
        return self.classifier(combined)
