from .custom import *
from torch.utils.data import DataLoader
import pdb
import torch

def make_data_loader(args):
    if args.test :
        benchmark_set = custom.CustomData(args, split='test')
        test_loader = DataLoader(benchmark_set, batch_size=args.test_batch_size, shuffle=False)
        return test_loader
    else:

        train_set = CustomData(args, split='train')
        
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        if not args.no_val:
            val_set = custom.CustomData(args, split='val')
            val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False)
            return train_loader, val_loader
        else:
            return train_loader