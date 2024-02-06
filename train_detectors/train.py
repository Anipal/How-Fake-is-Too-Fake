'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np 
import random
import wandb
import torchvision
from modelling import * 
import os
from tqdm import tqdm 
import argparse
from utils import progress_bar
from data import make_data_loader
import pdb
from tqdm import tqdm
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Training for DeepFake detection')
    #Main configs
    parser.add_argument('--task', type=str, default='training', help='type of task example training, stress testing')
    parser.add_argument('--exp', type=str, default='0', help='exp number')
    parser.add_argument('--nclass', type=int, default=2,help='num (default: 2)')
    parser.add_argument('--backbone', type=str, default='resnet50',choices=['resnet50', 'resnest','swin'],help='backbone name (default: resnet)')
    parser.add_argument('--dataset', type=str, default='custom', choices=['genimage', 'imagenet', 'custom'],help='dataset name (default: custom)')
    parser.add_argument('--loss-type', type=str, default='ce',choices=['ce', 'focal','dice','lovasz','cejac', 'jac', 'fojac', 'fodice', 'fodicece', 'foce', 'folov', 'celov', 'softnll', 'smoothfojac','smoothcejac' ,'fbeta', 'combo', 'ohem', 'tversky', 'cedice', 'ce_ohem'],help='loss func type (default: ce)')

    # # Image configs
    # parser.add_argument('--out-stride', type=int, default=16,help='network output stride (default: 8)')
    # parser.add_argument('--base-size', type=int, default=512,help='base image size')
    # parser.add_argument('--crop-size', type=int, default=512,help='crop image size')
    # parser.add_argument('--stitch_size', type=int, default=768,help='crop image size')
    # parser.add_argument('--tiling', action='store_true', default=False,help='finetuning on a different dataset')
    # parser.add_argument('--align', action='store_true', default=False, help='enables aligment')
    # parser.add_argument('--brightness_threshold', type=float, default=0.5,help='alpha (default: 0.9)')
    
    ##Batchnorm options
    parser.add_argument('--sync-bn', type=bool, default=None,help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,help='whether to freeze bn parameters (default: False)')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=4,metavar='N', help='input batch size for \training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=1,metavar='N', help='input batch size for \testing (default: auto)')

    # optimizer params
    parser.add_argument('--optimizer', type=str, default='adam',choices=['sgd', 'adam','adabound'],help='optimizer for the model: (default: sgd)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',help='learning rate (default: auto)')
    parser.add_argument('--momentum', type=float, default=0.9,metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-6,metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,help='whether use nesterov (default: False)')

    #LR Scheduler 
    parser.add_argument('--lr-scheduler', type=str, default='poly',choices=['poly', 'step', 'cos','exp','multistep'],help='lr scheduler mode: (default: poly)')
    parser.add_argument('--gamma', type=float, default=0.1,help='gamma (default: 0.1)')
    parser.add_argument('--power', type=float, default=1.0,help='power (default: 1.0)')
    parser.add_argument('--lr-step', type=int, default=10,help='lr step (default: 10)')
    parser.add_argument('--min-lr', type=float, default=1e-6,help='gamma (default: 0.0)')

    #Misc
    parser.add_argument('--seed', type=int, default=0, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--wandb_dir', type=str, default='../wandb',help='put the path to wandb run directory')

    # checkpoint
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--checkpoint', type=str, default=None,help='set the checkpoint name')

    # finetuning pre-trained models
    parser.add_argument('--pre', action='store_true', default=False,help='using pretrained weights')

    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,help='skip validation during training')
#     parser.add_argument('--flag', type=str, default='/media/ssd1/data/sanborn',help='put the path to root directory')

    ##test option
    parser.add_argument('--test', action='store_true', default=False,help='Run testing')
    parser.add_argument('--real', action='store_true', default=False,help='Run testing')
    parser.add_argument('--save_dir', type=str, default='../training_models/',help='put the path to pmap saving directory')

    #splits 
    parser.add_argument('--data_split', type=str, help='The dataset to be trained on',  default = None)
    parser.add_argument('--prompt', type=str, help='The prompt category to be trained on',  default = None)
   
    ##files
    parser.add_argument('--root_csv', type=str, required=True, help='<Required> Set flag',  default = None)
    parser.add_argument('--root_dir', type=str, required=True, default='/media/ssd1/data/sanborn',help='put the path to root directory')
    ############################################################################################################
    args = parser.parse_args()
    return args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

# Training
def train(epoch, net, criterion, optimizer, train_loader):
    print('\nEpoch: %d' % epoch)
    
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    
    for batch_idx, sample in enumerate(tqdm(train_loader)):

        inputs, name, targets = sample['image'], sample['image_name'],sample['label']
        inputs , targets = inputs.to(device), targets.to(device)
        targets = targets.unsqueeze(1)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets.float())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        wandb.log({"Running Train loss": loss.item(),"iteration": batch_idx+1})
        predicted = torch.where(outputs > 0.5, torch.tensor(1), torch.tensor(0))
        total += targets.size(0)
        # pdb.set_trace()
        correct += predicted.eq(targets).sum().item()

    return train_loss/(batch_idx+1), 100.*(correct/total)


def val(epoch, net, criterion, val_loader):
    global best_acc
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    # pdb.set_trace()
    if(args.test):
        wrong = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(val_loader)):

            inputs, name, targets = sample['image'], sample['image_name'], sample['label']
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            targets = targets.unsqueeze(1)
            loss = criterion(outputs, targets.float())

            val_loss += loss.item()
            
            predicted = torch.where(outputs > 0.5, torch.tensor(1), torch.tensor(0))
            total += targets.size(0)
            if(predicted.eq(targets).sum().item() == 0) and args.test:
                wrong.append(sample['image_name'])
            correct += predicted.eq(targets).sum().item()
    acc = 100.*(correct/total)
    if(args.test):
        return (val_loss/(batch_idx+1)), acc, wrong
    return (val_loss/(batch_idx+1)), acc

# def main():
# pdb.set_trace()
args = get_args()
if not(args.test):
    os.environ["WANDB_DIR"] = args.wandb_dir
    wandb.init(project=args.backbone ,config=args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0.0  # best test accuracy
start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
if(args.test):
    test_loader = make_data_loader(args)
elif (args.no_val):
    train_loader = make_data_loader(args)
else:
    train_loader, val_loader = make_data_loader(args)

# Model
print('==> Building model..')
net = return_model(args)
net = net.to(device)


if args.resume or args.test:
    # Load checkpoint.
    
    print('==> Loading checkpoint..')
    ckpt_dir = os.path.join(args.save_dir,'checkpoint')
    assert os.path.isdir(ckpt_dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(ckpt_dir,args.checkpoint))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = criterion(args)

if(not(args.test)):
    optimizer = return_optim(args, net)
    scheduler = set_scheduler(args, net, optimizer)
    for epoch in tqdm(range(start_epoch, args.epochs)):
            train_loss, train_acc = train(epoch, net, criterion, optimizer, train_loader)
            wandb.log({"train_loss": train_loss, "train_acc": train_acc,"epoch": epoch+1})
            if(((epoch+1)% args.eval_interval) == 0) and not(args.no_val):
                val_loss, val_acc = val(epoch, net, criterion, val_loader)
                progress_bar(epoch + 1, args.epochs, train_loss, val_loss, train_acc, val_acc)
                wandb.log({"val_loss": val_loss, "val_acc": val_acc,"epoch": epoch+1})

                if val_acc > best_acc:
                    print('Saving..')
                    state = {
                        'net': net.state_dict(),
                        'acc': val_acc,
                        'epoch': epoch,
                    }
                
                    if(args.prompt):
                        if not(os.path.exists(os.path.join(args.save_dir,'checkpoint',args.backbone,args.prompt,args.exp))):
                            os.makedirs(os.path.join(args.save_dir,'checkpoint',args.backbone,args.prompt,args.exp))
                        save_path = os.path.join(args.save_dir,'checkpoint',args.backbone,args.prompt,args.exp,('ckpt_epoch_'+str(epoch)+'_acc_'+str(round(val_acc, 2))+'.pth'))
                    elif(args.data_split):
                        if not(os.path.exists(os.path.join(args.save_dir,'checkpoint',args.backbone,args.data_split,args.exp))):
                            os.makedirs(os.path.join(args.save_dir,'checkpoint',args.backbone,args.data_split,args.exp))
                        save_path = os.path.join(args.save_dir,'checkpoint',args.backbone,args.data_split,args.exp,('ckpt_epoch_'+str(epoch)+'_acc_'+str(round(val_acc, 2))+'.pth'))
                    else:
                        if not(os.path.exists(os.path.join(args.save_dir,'checkpoint',args.backbone,'all',args.exp))):
                            os.makedirs(os.path.join(args.save_dir,'checkpoint',args.backbone,'all',args.exp))
                        save_path = os.path.join(args.save_dir,'checkpoint',args.backbone,'all',args.exp,('ckpt_epoch_'+str(epoch)+'_acc_'+str(round(val_acc, 2))+'.pth'))
                    torch.save(state, save_path)
                    best_acc = val_acc
            scheduler.step()
else:
    # pdb.set_trace()
    test_loss, test_acc, wrong = val(1, net, criterion, test_loader)
    print('TEST LOSS: ',test_loss, 'TEST ACC :', test_acc, 'NUM_SAMPLES :', len(test_loader))
    
    text = f"TEST LOSS: {test_loss}, TEST ACC: {test_acc}, 'NUM_SAMPLES :', {len(test_loader)}\n"
    text += "Misclassified samples:\n"
    for sample in wrong:
        text += f"{sample}\n"

    if(args.real):
        txt = '_real.txt'
    else:
        txt = '.txt'
    if args.prompt:
        file_path = os.path.join(os.path.dirname(os.path.join(ckpt_dir,args.checkpoint)),(args.prompt+txt))
    elif(args.data_split):
        file_path = os.path.join(os.path.dirname(os.path.join(ckpt_dir,args.checkpoint)),(args.data_split+txt))
    else:
        file_path = os.path.join(os.path.dirname(os.path.join(ckpt_dir,args.checkpoint)),('all'+txt))

    with open(file_path, "w") as file:
        file.write(text)

