import torch.nn as nn 

def criterion(args):
    if(args.loss_type == 'ce'):
        loss_ = nn.BCELoss()
    return loss_