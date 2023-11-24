import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageOps
import ipdb
import json
import ast


TARGET_SIZE = 5000
MASKS_PER_IMAGE = 3
ADE20K = "/data/jkruk3/ADE20K_People_256"
CITYSCAPES = "/data/jkruk3/CityScapes_256"
CELEBA = "/data/jkruk3/CelebAHQ_256"
HPD = "/data/jkruk3/HumanParsing_256"

DEST_DIR = "/data/jkruk3/Diffm_FakeImage_Dataset"
DEST_NAME = "Diffm_FakeImage_Dataset"
OUT_FILE = "/data/jkruk3/Diffm_FakeImage_Dataset/metadata_5k.json"
if not os.path.exists(DEST_DIR):
    os.mkdir(DEST_DIR)

try:
    with open(os.path.join(ADE20K, "metadata.json"), "r") as ff:
        ade20k_meta = json.load(ff)
    with open(os.path.join(CITYSCAPES, "metadata.json"), "r") as ff:
        cityscapes_meta = json.load(ff)
    with open(os.path.join(CELEBA, "metadata.json"), "r") as ff:
        celeba_meta = json.load(ff)
    hpd_meta = pd.read_csv(os.path.join(HPD, 'metadata.csv'))

except:
    ADE20K = "/data/jkruk3/Diffm_FakeImage_Dataset/ADE20K_People_256"
    CITYSCAPES = "/data/jkruk3/Diffm_FakeImage_Dataset/CityScapes_256"
    CELEBA = "/data/jkruk3/Diffm_FakeImage_Dataset/CelebAHQ_256"
    HPD = "/data/jkruk3/Diffm_FakeImage_Dataset/HumanParsing_256"

    with open(os.path.join(ADE20K, "metadata.json"), "r") as ff:
        ade20k_meta = json.load(ff)
    with open(os.path.join(CITYSCAPES, "metadata.json"), "r") as ff:
        cityscapes_meta = json.load(ff)
    with open(os.path.join(CELEBA, "metadata.json"), "r") as ff:
        celeba_meta = json.load(ff)
    hpd_meta = pd.read_csv(os.path.join(HPD, 'metadata.csv'))


# Inspect individual dataset sizes:
print(f"Size of ADE20k processed benchmark:   {len(ade20k_meta.keys())}")
print(f"Size of CITYSCAPES processed benchmark:   {len(cityscapes_meta.keys())}")
print(f"Size of CELEBA-HQ processed benchmark:   {len(celeba_meta.keys())}")
print(f"Size of HUMAN-PARSING processed benchmark:   {len(hpd_meta)}")

# TODO: Standardized dataset metadata:
full_data = {}
ade_img_dir = "ADE20K_People_256/images"


for key, meta in ade20k_meta.items():
    meta_refact = {
        "image_path": os.path.join(
            DEST_NAME, meta["image"].replace("/data/jkruk3/", "")
        ),
        "entities": meta["elements"],
        "dataset": "ADE20K",
    }

    objects = []
    for obj in meta["metadata"]["object"]:
        o_ = {
            "id": obj["instance_mask"].replace(".png", ""),
            "mask_path": os.path.join(DEST_NAME, ade_img_dir, obj["instance_mask"]),
            "name": obj["name"],
            "ratio": obj["area_ratio"],
        }
        objects.append(o_)
    meta_refact["objects"] = objects
    full_data[key] = meta_refact

print('completed ade20k')


for key, meta in cityscapes_meta.items():
    entities = list(set([x["entity"] for x in meta["objects"]]))
    meta_refact = {
        "image_path": os.path.join(
            DEST_NAME, meta["image_path"].replace("/data/jkruk3/", "")
        ),
        "entities": entities,
        "dataset": "CityScapes",
    }

    objects = []
    for obj in meta["objects"]:
        o_ = {
            "id": obj["id"],
            "mask_path": os.path.join(
                DEST_NAME, obj["mask_path"].replace("/data/jkruk3/", "")
            ),
            "name": obj["entity"],
            "ratio": obj["ratio"],
        }
        objects.append(o_)
    meta_refact["objects"] = objects
    full_data[key] = meta_refact

print('completed cityscapes')


remaining_scale = int((TARGET_SIZE - len(full_data)) / 2)

count = 0
for key, meta in celeba_meta.items():
    if count < remaining_scale:
        entities = list(set([x["entity"] for x in meta["objects"]]))
        meta_refact = {
            "image_path": os.path.join(
                DEST_NAME, meta["image_path"].replace("/data/jkruk3/", "")
            ),
            "entities": entities,
            "dataset": "CelebA-HQ",
        }

        objects = []
        for obj in meta["objects"]:
            o_ = {
                "id": obj["image_id"],
                "mask_path": os.path.join(
                    DEST_NAME, obj["mask_path"].replace("/data/jkruk3/", "")
                ),
                "name": obj["entity"],
                "ratio": obj["ratio"],
            }
            objects.append(o_)
        meta_refact["objects"] = objects
        full_data[f"CelebA_{key}"] = meta_refact
        count += 1

print('completed cityscapes')


count = 0
for i, row in hpd_meta.iterrows():
    if count < remaining_scale:
        masks = ast.literal_eval(row["masks"])
        entities = list(set([x["element"] for x in masks]))
        meta_refact = {
            "image_path": os.path.join(
                DEST_NAME, row["img_path"].replace("/content/drive/MyDrive/MLwLS Fall23/", "")
            ),
            "entities": entities,
            "dataset": "HumanParsing",
        }

        objects = []
        for obj in masks:
            id = obj['mask_path'][obj['mask_path'].rfind('/')+1:].replace('.png', '')
            o_ = {
                "id": id,
                "mask_path": os.path.join(
                    DEST_NAME, obj["mask_path"].replace("/content/drive/MyDrive/MLwLS Fall23/", "")
                ),
                "name": obj["element"],
                "ratio": obj["area_ration"],
            }
            objects.append(o_)
        meta_refact["objects"] = objects
        full_data[row.img_id] = meta_refact
        count += 1

print('completed human parsing')

with open(OUT_FILE, 'w') as jf:
    json.dump(full_data, jf)
print(f"Final size of the dataset:        {len(full_data)}")
print(f"Consolidated metadata saved at:   {OUT_FILE}")

