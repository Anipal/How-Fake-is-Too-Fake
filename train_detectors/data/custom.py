import os
import math
import numpy as np
import pdb
import torch
import cv2
import random
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F 
import glob
import pandas as pd 


class CustomData(Dataset):
    """
    Custom dataset loader 
    """

    def __init__(self,args,split='train'):
        """
        :param split: train/val/test
        """

        super().__init__()
        self.split = split
        self.args = args
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((224,224))
            ])
        self.images,self.labels = self._get_image()


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
  
        # self._newindex = index%len(self.images)
#         print(self._newindex)
        _img,_label, _img_name= self._make_img_gt_point_pair(index)
        sample = {'image': _img,'label': _label,'image_name':_img_name}
        return sample

    def _get_image(self):

        df = pd.read_csv(self.args.root_csv)
 
        df['image_path'] = self.args.root_dir + df['image_path']

        data_split = self.args.data_split
        prompt = self.args.prompt
        real = self.args.real
        if not(data_split == None):
            if(self.split == 'test') and not(real):
                df = df[df['label'] =='fake']
            df = df[(df['split'] == self.split) & (df['dataset'] == self.args.data_split)]
            # print('HERE 1')
        elif not(prompt == None):
            if(self.split == 'test') and not(real):
                df = df[(df['split'] == self.split) & ((df['pertrubation_type'] == self.args.prompt))]
            else:
                df = df[(df['split'] == self.split) & ((df['pertrubation_type'] == self.args.prompt) | (df['label'] == 'real'))]
            # print('HERE 2')
        else:
            df = df[(df['split'] == self.split)]
            # print('HERE 3')
        
        image_path = df['image_path']

        labels = df['label']
        label_encoding = {'real': 0, 'fake': 1}
        num_classes = len(label_encoding)
        # print(num_classes)
        df['label_index'] = df['label'].map(label_encoding)
        labels = torch.tensor(df['label_index'].values)
        # if(self.split == 'train'):
        #     print('IDX 1',image_path.iloc[2368],image_path.iloc[3236])
        # labels_one_hot = torch.zeros(len(label_indices_tensor), num_classes)
        # labels_one_hot.scatter_(1, label_indices_tensor.unsqueeze(1), 1)

        return image_path, labels



    def _make_img_gt_point_pair(self, index):
        # print('IDX', index)
        # print('TGT',self.labels[index])
        # print('IMG',self.images[index])
        img_name = self.images.iloc[index]
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[index]
        img = self.transform(img)
        return img,label,img_name
    def __str__(self):
        pass
