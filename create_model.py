
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt

def create_effnetB2_model(out_features= 3, DEVICE = "cpu"):
  effnetB2_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
  effnetB2_transforms = effnetB2_weights.transforms()

  effnet_B_model = torchvision.models.efficientnet_b2(weights = effnetB2_weights).to(DEVICE)

  for params in effnet_B_model.parameters():
      params.requires_grad = False

  effnet_B_model.classifier = nn.Sequential(
      nn.Dropout(p = 0.2, inplace = True),
      nn.Linear(in_features = 1408, out_features = out_features, bias = True)
  ).to(DEVICE)

  return effnet_B_model, effnetB2_transforms
