"""
ResNet9 architecture used for plant disease image classification
(trained on the PlantVillage-style 38-class dataset).

This file only defines the network. The trained weights file
`plant_disease_model.pth` must be downloaded separately (see README /
Google Drive link) and placed inside the `models/` folder, because
model binaries are not committed to the git repository.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class SimpleResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return x + out


class ResNet9(nn.Module):
    """Input: 3 x 256 x 256 RGB leaf image. Output: logits for `num_classes`."""

    def __init__(self, in_channels=3, num_classes=38):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = SimpleResidualBlock(128)

        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = SimpleResidualBlock(512)

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out if False else self.res1(out)  # residual add is inside block
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out)
        out = self.classifier(out)
        return out


def load_model(weights_path: str, num_classes: int = 38, device: str = "cpu"):
    """Loads the ResNet9 model with pretrained weights. Returns None if weights are missing."""
    import os
    model = ResNet9(num_classes=num_classes)
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model
    return None


def predict_image(image_bytes, model, classes, device="cpu"):
    """Runs a single PIL-loadable image (bytes) through the model and returns the predicted class label."""
    from PIL import Image
    from torchvision import transforms
    import io

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    return classes[pred_idx.item()], round(conf.item() * 100, 2)
