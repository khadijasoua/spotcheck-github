"""Shared data loading, preprocessing, model construction and inference for SpotCheck."""

import torch
from medmnist import INFO, DermaMNIST
from torch import nn
from torchvision import models, transforms

SIZE = 128  # image side length in pixels (DermaMNIST offers 28, 64, 128, 224)

# Class names in label order (0-6), e.g. "melanoma".
NAMES = INFO["dermamnist"]["label"]
CLASS_NAMES = [NAMES[str(i)] for i in range(len(NAMES))]
MEL = CLASS_NAMES.index("melanoma")

# ImageNet normalisation, matching the pretrained ResNet-18 weights.
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Training-time augmentation: lesions are orientation-invariant.
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    to_tensor,
])


def get_data(split, train_mode=False):
    """Load a DermaMNIST split ('train', 'val' or 'test'); augmented if train_mode."""
    tf = augment if train_mode else to_tensor
    return DermaMNIST(split=split, download=True, size=SIZE, transform=tf)


def build_model(pretrained=True):
    """ResNet-18 with its 1,000-class head replaced by a 7-class linear layer.

    Use pretrained=True for training (ImageNet weights) and pretrained=False
    when loading our own fine-tuned weights from a .pt file.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


@torch.no_grad()
def get_probs(model, dataset):
    """Return softmax probabilities (N x 7) and true labels (N) for a dataset."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=64)
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        all_probs.append(torch.softmax(model(x), dim=1))
        all_labels.append(y.squeeze(1).long())
    return torch.cat(all_probs), torch.cat(all_labels)
