import argparse
import json
import os
from PIL import Image
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import pdb
class CustomImageFolder(Dataset):
    def __init__(self,data_dir, json_path, dataset='all'):
        # super(CustomImageFolder, self).__init__(data_dir)
        self.filenames = []
        self.cls = []
        self.data_dir = data_dir
        self.json = json.load(open(json_path))
        # pdb.set_trace()
        if(dataset == 'all'):
            for key, val in (self.json).items():
                
                self.filenames.append(val['image_path'])
                self.cls.append(val['class'][0] if isinstance(val['class'],list) else val['class'])
        else:
            for key, val in (self.json).items():
                if(val['dataset'] == dataset):
                    self.filenames.append(val['image_path'])
                    self.cls.append(val['class'][0] if isinstance(val['class'],list) else val['class'])

        
        # self.filenames = sorted(os.listdir(self.data_dir))
        # self.cls = [data_dir.split('/')[-1]]*len(self.filenames)
        
    def __getitem__(self, ind):
        filename = self.filenames[ind]
        img_path = os.path.join(self.data_dir, filename)
        clsname = self.cls[ind]      
        # print(clsname, ':', img_path) 
        return img_path, clsname

    def __len__(self):
        return len(self.filenames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        default="/srv/hoffman-lab/share4/datasets/ADE20K_People/images/orig/person",
        help="Path to custom dataset",
    )
    args = parser.parse_args()
    # pdb.set_trace()
    dset = CustomImageFolder(args.img_dir)
    print(dset[0])