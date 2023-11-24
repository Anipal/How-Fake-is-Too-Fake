import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageOps
import random
import shutil
import ipdb
import json
import ast


MASKS_PER_IMAGE = 3
ROOT_DIR = "/data/jkruk3/"
ORG_NAME = "Diffm_FakeImage_Dataset"
DEST_NAME = "Diffm_FakeImage_Dataset_5k"

METADATA = "/data/jkruk3/Diffm_FakeImage_Dataset_5k/metadata_5k.json"
IMG_MASK_META = "/data/jkruk3/Diffm_FakeImage_Dataset_5k/img_masks_5k.json"

ADE20K = "/data/jkruk3/Diffm_FakeImage_Dataset_5k/ADE20K_People_256"
CITYSCAPES = "/data/jkruk3/Diffm_FakeImage_Dataset_5k/CityScapes_256"
CELEBA = "/data/jkruk3/Diffm_FakeImage_Dataset_5k/CelebAHQ_256"
HPD = "/data/jkruk3/Diffm_FakeImage_Dataset_5k/HumanParsing_256"

if not os.path.exists(ADE20K):
    os.mkdir(ADE20K)
if not os.path.exists(CITYSCAPES):
    os.mkdir(CITYSCAPES)
if not os.path.exists(CELEBA):
    os.mkdir(CELEBA)
if not os.path.exists(HPD):
    os.mkdir(HPD)
if not os.path.exists(METADATA):
    org_path = METADATA.replace(DEST_NAME, ORG_NAME)
    shutil.copy(org_path, METADATA)

def download_file(org_path, dest_path):
    try:
        shutil.copy(org_path, dest_path)
    except:
        root_ = dest_path[:dest_path.rfind('/')]
        os.mkdir(root_)
        shutil.copy(org_path, dest_path)

# ___________________________________________________________________ #


with open(METADATA, "r") as ff:
    full_data = json.load(ff)

for key, meta in tqdm(full_data.items(), total=len(full_data)):
    org_path = os.path.join(ROOT_DIR, meta['image_path'])
    dest_path = os.path.join(ROOT_DIR, meta['image_path'].replace(ORG_NAME, DEST_NAME))
    download_file(org_path, dest_path)

    for obj in meta["objects"]:
        org_path = os.path.join(ROOT_DIR, obj['mask_path'])
        dest_path = os.path.join(ROOT_DIR, obj['mask_path'].replace(ORG_NAME, DEST_NAME))
        download_file(org_path, dest_path)
    
print(f"All image files transfered to:   {os.path.join(ROOT_DIR, DEST_NAME)}")

# isoldating certain amounts of masks per image:
total_pairs = 0
img_mask_pairs = {}
for key, meta in tqdm(full_data.items(), total=len(full_data)):
    try:
        select_masks = random.sample(meta["objects"], MASKS_PER_IMAGE)
    except ValueError:
        select_masks = meta["objects"]

    total_pairs += len(select_masks)
    meta_select = meta.copy()
    meta_select["objects"] = select_masks
    img_mask_pairs[key] = meta_select


with open(IMG_MASK_META, "w") as ff:
    json.dump(img_mask_pairs, ff)
print(f"Image-mask pair metadata saved at:   {IMG_MASK_META}")
print(f"Number of unique image-mask pairs:   {total_pairs}")
