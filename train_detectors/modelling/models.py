import torch
from torchvision import models

def return_model(args):
    if args.backbone == 'resnet50':
        model = models.resnet50(pretrained=args.pre)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )
    elif args.backbone == 'swin':
        model = models.swin_t(pretrained=args.pre, num_classes=num_classes) 
        model.head = nn.Linear(model.head.in_features, 1)
        model.head.act = nn.Sigmoid()

    return model 