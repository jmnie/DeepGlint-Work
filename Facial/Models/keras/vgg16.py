import torch
from torchvision import models
from torch import nn


def vgg16_fine_tune():
    model_ft = models.vgg16(pretrained=True)

    for param in model_ft.parameters():
        param.requires_grad = False

    fc_features = model_ft.fc.in_features # Extract the number of the fully connected layer
    model_ft.fc = nn.Linear(fc_features,7)

    return model_ft