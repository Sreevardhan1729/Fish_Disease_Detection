import torch
from torch import nn
import torchvision
from typing import List
def create_model(num_classes:int):
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    model = torchvision.models.vit_b_16(weights=weights)
    for param in model.parameters():
        param.requires_grad=False
    model.heads = nn.Sequential(
        nn.Linear(in_features=768,out_features=num_classes)
    )
    return model
