from datasets import Dataset, DatasetDict, Image, Value
import glob 
import os
import pdb 
from tqdm import tqdm 
root_dir = '/srv/hoffman-lab/share4/apal72/half-truths/data/Half_Truths_Dataset'

datasets = [l.split('/')[-1] for l in glob.glob(root_dir +'/images/*')]
dict_ = {}




def create_dataset(image_paths, label_paths, img_names, mask_names):
    dataset = Dataset.from_dict({"image": image_paths,
                                "label": label_paths ,
                                "img_name": img_names,
                                "mask_name":mask_names,})
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Image())
    dataset = dataset.cast_column("img_name", Value(dtype='string', id=None))
    dataset = dataset.cast_column("mask_name", Value(dtype='string', id=None))
    return dataset


for d in datasets:
    images = glob.glob(root_dir +'/images/'+d +'/*')
    masks = glob.glob(root_dir +'/masks/'+d +'/*')
    # mask_names = [os.path.splitext(mask_name.split('/')[-1])[0] for mask_name in masks]
    final_images = []
    final_masks = []
    img_names =[]
    mask_names=[]
    for img in tqdm(images, total =len(images)):
        if(d == 'CelebAHQ'):
            img_name = "{:0>5}".format(os.path.splitext(img.split('/')[-1])[0])
        else:
            img_name = os.path.splitext(img.split('/')[-1])[0]

        img_count = sum(s.count(img_name) for s in masks)
        mask_counts = [string for string in masks if img_name in string]
        img_names.extend([img_name]*img_count)
        mask_names.extend([os.path.splitext(mask_name.split('/')[-1])[0] for mask_name in mask_counts])
        final_masks.extend(mask_counts)
        final_images.extend([img]*img_count)


    # pdb.set_trace()
    data = create_dataset(final_images, final_masks, img_names,mask_names)
    dict_[d] = data
pdb.set_trace()
dataset = DatasetDict(dict_)
dataset.push_to_hub("Half-Truths-Project/base-datasets-2", private=True)
