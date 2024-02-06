import torch
from torchvision import models
import torch.nn as nn

from .loss import *
from .optimizer import *
from .scheduler import *



def return_model(args):
    if args.backbone == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )
    elif args.backbone == 'swin':
        HUB_URL = "SharanSMenon/swin-transformer-hub"
        MODEL_NAME = "swin_tiny_patch4_window7_224"
        model = torch.hub.load(HUB_URL, MODEL_NAME, pretrained=True)
        n_inputs = model.head.in_features
        print('N',n_inputs)
        model.head = nn.Sequential(
            nn.Linear(n_inputs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
            )
        model.head.act = nn.Sigmoid()

    return model 