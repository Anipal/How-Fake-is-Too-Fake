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

DATA_DIR = "/data/jkruk3/ADE20K_People_256/"
META_DATA = "/data/jkruk3/ADE20K_People_256/metadata.json"
IMG_DIR = "/data/jkruk3/ADE20K_People_256/images"

DEST_MASK_DIR = "/nethome/jkruk3/how_fake/RePaint/data/datasets/ade20k_keep_masks_d2"
DEST_IMG_DIR = "/nethome/jkruk3/how_fake/RePaint/data/datasets/ade20k_images_d2"
DEST_META_DIR = "/nethome/jkruk3/how_fake/RePaint/data/datasets/ade20k_metadata"
if not os.path.exists(DEST_MASK_DIR):
    os.mkdir(DEST_MASK_DIR)
if not os.path.exists(DEST_IMG_DIR):
    os.mkdir(DEST_IMG_DIR)
if not os.path.exists(DEST_META_DIR):
    os.mkdir(DEST_META_DIR)


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


# TODO: Scheme for changing multiple masked objects at once:
#   - semantically similar objects
#   - requires mask merging (pairwise addition on pixel values)

if __name__ == "__main__":

    # FOR DEMO: generate a few examples with masked items:
    #   - person
    #   - head
    #   - car
    #   - window
    #   - wall
    #   - tree

    with open(META_DATA, "r") as jf:
        data = json.load(jf)

    all_elements = []
    for _, value in data.items():
        all_elements = list(set(all_elements + value["elements"]))
    ipdb.set_trace()

    # elements = ["person", "head", "car", "window", "wall", "tree"]
    elements = ["left arm", "right leg", "face", "hand", "wall", "plants", "river", "blackboard"]
    examples = 3

    select = []
    for elem in elements:
        select_ = get_image_mask_pairs(data, elem, examples)
        select = select + select_

    print(len(select))
    meta_path = os.path.join(DEST_META_DIR, "demo_metadata_v1.json")
    with open(meta_path, "w") as outfile:
        json.dump(select, outfile)

