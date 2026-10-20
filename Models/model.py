import torch.nn as nn
from torchvision import models


def create_model(num_classes=6):
    """
    Create a ResNet18 model for steel defect classification.

    Args:
        num_classes: Number of defect categories.

    Returns:
        ResNet18 model with a modified classification layer.
    """

    # Create ResNet18 without downloading pretrained weights
    model = models.resnet18(weights=None)

    # Get the number of features entering the final layer
    num_features = model.fc.in_features

    # Replace the original classifier
    model.fc = nn.Linear(
        num_features,
        num_classes
    )

    return model
