"""
This script takes a processed semantic segmentation data and
prepares it for input to Repaint.

Repaint requires that images and masks be in seperate
directories, but have the same name. 

This script will also produce an accompanying json file that
will describe the objects masked, the mask size, and objects
present in the base, real, image.
"""

import os
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import ipdb
import json
import geopandas as gpd
from shapely.geometry import Polygon, mapping
from shapely import affinity
from PIL import Image
import PIL.ImageOps
import shutil
import random

# TODO: There may be cases in which I would want to merge masks
# However, this would only work for certain semantic categories ...
MASKS_PER_IMAGE = 3
DATA_DIR = "/data/jkruk3/Diffm_FakeImage_Dataset_5k"
META_DATA = "/data/jkruk3/Diffm_FakeImage_Dataset_5k/img_masks_5k.json"

# DEST_META_DIR = "/nethome/jkruk3/how_fake/RePaint/data/datasets/ade20k_metadata"
DEST_IMG_DIR = "/data/jkruk3/RePaint/input/images"
DEST_MASK_DIR = "/data/jkruk3/RePaint/input/keep_masks"

if not os.path.exists(DEST_MASK_DIR):
    os.mkdir(DEST_MASK_DIR)
if not os.path.exists(DEST_IMG_DIR):
    os.mkdir(DEST_IMG_DIR)
# if not os.path.exists(DEST_META_DIR):
#     os.mkdir(DEST_META_DIR)


# NOTE: Scheme for changing one masked object at a time:
def get_image_mask_pairs(data, elem, examples):

    count = 0
    select = []
    for key, value in data.items():
        if elem in value["elements"] and count < examples:

            new_key = key + f"_{count}"
            img_path = value["image"]
            dest_img_path = os.path.join(DEST_IMG_DIR, new_key + ".png")
            shutil.copy(img_path, dest_img_path)

            d_ = {
                "name": new_key,
                "image": dest_img_path,
                "dataset": "ADE20k_People_256",
                "origin_name": key,
                "image_elements": value["elements"],
            }

            objects = value["metadata"]["object"]
            random.shuffle(objects)

            for obj in objects:
                if obj["name"] == elem:
                    mask_path = os.path.join(IMG_DIR, obj["instance_mask"])
                    dest_mask_path = os.path.join(DEST_MASK_DIR, new_key + ".png")
                    shutil.copy(mask_path, dest_mask_path)

                    d_["mask"] = dest_mask_path
                    d_["masked_element"] = elem
                    d_["area_ratio"] = obj["area_ratio"]
                    d_["origin_instance_mask"] = [obj["instance_mask"]]

                    break

            select.append(d_)
            count += 1
    return select


def download_image_mask_pairs(data, batches=1):

    all_pair_ids = []
    processed_meta = {}
    count = 0
    for key, value in tqdm(data.items(), total=len(data)):

        # this piece of code fixes a file_path error in metadata:
        ORG_ROOT = "Diffm_FakeImage_Dataset"
        NEW_ROOT = "Diffm_FakeImage_Dataset_5k"

        img_path = value['image_path']
        if ORG_ROOT in img_path:
            img_path = img_path[img_path.find('/')+1:]
        value['image_path'] = img_path
        
        root_img = Image.open(os.path.join(DATA_DIR, img_path))
        for obj in value["objects"]:

            # this piece of code fixes a file_path error in metadata:
            mask_path = obj["mask_path"]
            if ORG_ROOT in mask_path:
                mask_path = mask_path[mask_path.find('/')+1:]
            obj['mask_path'] = mask_path

            # saving image-mask pairs for inpaint model input
            pair_name = str(count).zfill(5)
            obj['inpaint_id'] = pair_name
            mask_img = Image.open(os.path.join(DATA_DIR, mask_path))

            if batches > 1:
                files_per_batch = int((len(data)* MASKS_PER_IMAGE) / batches) + 1
                batch_num = int(count / files_per_batch)
                BATCH_DIR = f"batch_{batch_num}"
                if not os.path.exists(os.path.join(DEST_MASK_DIR, BATCH_DIR)):
                    os.mkdir(os.path.join(DEST_MASK_DIR, BATCH_DIR))
                if not os.path.exists(os.path.join(DEST_IMG_DIR, BATCH_DIR)):
                    os.mkdir(os.path.join(DEST_IMG_DIR, BATCH_DIR))

                mask_img.save(os.path.join(DEST_MASK_DIR, BATCH_DIR, f"{pair_name}.png"))
                root_img.save(os.path.join(DEST_IMG_DIR, BATCH_DIR, f"{pair_name}.png"))
            else:
                mask_img.save(os.path.join(DEST_MASK_DIR, f"{pair_name}.png"))
                root_img.save(os.path.join(DEST_IMG_DIR, f"{pair_name}.png"))

            all_pair_ids.append(pair_name)
            count += 1
        
        processed_meta[key] = value
    return processed_meta, all_pair_ids


if __name__ == "__main__":

    with open(META_DATA, "r") as jf:
        data = json.load(jf)
    
    processed_meta, all_pair_ids = download_image_mask_pairs(
        data, 
        batches=10)
    
    with open(META_DATA, "w") as jf:
        json.dump(processed_meta, jf)

    print(f"{len(all_pair_ids)} image-mask pairs prepared for inpainting at:   {DEST_IMG_DIR[:DEST_IMG_DIR.rfind('/')]}")    

