import os
import numpy as np
import pandas as pd
import datasets
from datasets import load_dataset
from tqdm.notebook import tqdm
import transformers
from PIL import Image, ImageOps
import ipdb
import json


# [ADJUST] Output file specifications:
CELEB_IMG_DIR = "/content/drive/MyDrive/celebahq_masks/CelebAMask-HQ/CelebA-HQ-img"
CELEB_MASK_DIR = (
    "/content/drive/MyDrive/celebahq_masks/CelebAMask-HQ/CelebAMask-HQ-mask-anno"
)

OUTPUT_DIR = "/content/drive/MyDrive/CelebAHQ_256"
IMG_DIR = "/content/drive/MyDrive/CelebAHQ_256/images"
MASK_DIR = "/content/drive/MyDrive/CelebAHQ_256/masks"
dest_size = 256


def center_crop(im, new_size):
    width, height = im.size  # Get dimensions

    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


def download_img(img, str_path, dest_dir):
    file_n = str_path[str_path.rfind("/") + 1 :]
    dest_n = os.path.join(dest_dir, file_n)
    img.save(dest_n)
    return dest_n


def download_mask(img, str_path, dest_dir):
    file_n = str_path + ".jpg"
    dest_n = os.path.join(dest_dir, file_n)
    img.save(dest_n)
    return dest_n


def verify_mask(mask, threshold=0.005):
    """
    Verify that mask has not been removed during
    smartcrop and rescale. HPD masks are not "clean",
    so some pixels remain outside of mask. Mask
    needs to have SA ratio to image of > 5%.
    """

    h, w = mask.size
    mask_ = mask.convert("1")
    img_arr = np.asarray(mask_)

    img_sa = h * w
    mask_sa = abs(img_sa - np.count_nonzero(img_arr))
    ratio = mask_sa / img_sa

    # ipdb.set_trace()
    return ratio, ratio > threshold


# _______________________________________________________ #

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
if not os.path.exists(IMG_DIR):
    os.mkdir(IMG_DIR)
if not os.path.exists(MASK_DIR):
    os.mkdir(MASK_DIR)

# Loading dataset images:
celeb_img_files = os.listdir(CELEB_IMG_DIR)

# Loading dataset masks:
celeb_anno_files = []
for root, dirs, files in os.walk(CELEB_MASK_DIR):
    for name in files:
        if name.endswith(".png"):
            celeb_anno_files.append(os.path.join(root, name))

mask_img_ids = []
for pth in celeb_anno_files:
    name = pth[pth.rfind("/") + 1 :]
    name = name[: name.find("_")]
    mask_img_ids.append(name)

mask_img_ids = list(set(mask_img_ids))
print(f"Number of images with mask information:   {len(mask_img_ids)}")

meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
ids_not_found = []

# checking if metadata file exists and loading:
if os.path.exists(meta_path):
    with open(meta_path, "r") as jf:
        data_dict = json.load(jf)
else:
    data_dict = {}

# processing images and masks:
for id in tqdm(mask_img_ids, total=len(mask_img_ids)):
    id_ = str(int(id)) + ".jpg"

    img_match = [im for im in celeb_img_files if im == id_]
    if len(img_match) != 1:
        ids_not_found.append(id_)
        print(f"image not found:   {id}")
        # ipdb.set_trace()
        continue

    img_pth = os.path.join(CELEB_IMG_DIR, img_match[0])
    file_n = img_pth[img_pth.rfind("/") + 1 :]
    dest_n = os.path.join(IMG_DIR, file_n)

    if not os.path.exists(dest_n) and id not in data_dict.keys():
        img = Image.open(img_pth)
        masks = [im for im in celeb_anno_files if id in im]
        meta = {"src_image_path": img_pth}

        w, h = img.size
        if w == h:
            # processing & downloading image
            img = img.resize((dest_size, dest_size))
            dest_pth = download_img(img, img_pth, IMG_DIR)
            meta["image_path"] = dest_pth

            # processing masks & downloading masks:
            processed_masks = []
            for m in masks:
                fn = m[m.rfind("/") + 1 : m.rfind(".")]
                entity = fn[fn.rfind("_") + 1 :]
                id = fn[: fn.find("_")]

                if "_l_" in fn and "lip" in entity:
                    entity = f"lower_{entity}"
                elif "_u_" in fn:
                    entity = f"upper_{entity}"
                elif "_l_" in fn:
                    entity = f"left_{entity}"
                elif "_r_" in fn:
                    entity = f"right_{entity}"

                m_img = Image.open(m)
                m_img = m_img.resize((dest_size, dest_size))
                m_img = ImageOps.invert(m_img)
                # display(m_img)
                # ipdb.set_trace()
                ratio, threshold_passed = verify_mask(m_img)

                if threshold_passed:
                    mask_path = download_mask(m_img, fn, MASK_DIR)
                    m_ = {
                        "image_id": id,
                        "entity": entity,
                        "mask_path": mask_path,
                        "ratio": ratio,
                    }
                    processed_masks.append(m_)

        else:
            # get new image size dimentions
            llen = h if h < w else w
            scale_ratio = dest_size / llen
            if h < w:
                h_new = dest_size
                w_new = scale_ratio * w
            else:
                h_new = scale_ratio * w
                w_new = dest_size

            # processing & downloading image
            img = img.resize((int(w_new), int(h_new)))
            img = center_crop(img, dest_size)
            dest_pth = download_img(img, img_pth, IMG_DIR)
            meta["image_path"] = dest_pth

            # processing masks & downloading masks:
            processed_masks = []
            for m in masks:
                fn = m[m.rfind("/") + 1 : m.rfind(".")]
                entity = fn[fn.rfind("_") + 1 :]
                id = fn[: fn.find("_")]

                if "_l_" in fn and "lip" in entity:
                    entity = f"lower_{entity}"
                elif "_u_" in fn:
                    entity = f"upper_{entity}"
                elif "_l_" in fn:
                    entity = f"left_{entity}"
                elif "_r_" in fn:
                    entity = f"right_{entity}"

                m_img = Image.open(m)
                m_img = m_img.resize((dest_size, dest_size))
                m_img = center_crop(m_img, dest_size)
                m_img = ImageOps.invert(m_img)
                # display(m_img)
                # ipdb.set_trace()
                ratio, threshold_passed = verify_mask(m_img)

                if threshold_passed:
                    mask_path = download_mask(m_img, fn, MASK_DIR)
                    m_ = {
                        "image_id": id,
                        "entity": entity,
                        "mask_path": mask_path,
                        "ratio": ratio,
                    }
                    processed_masks.append(m_)

        meta["objects"] = processed_masks
        data_dict[id] = meta

        with open(meta_path, "w") as jf:
            json.dump(data_dict, jf)

# final save for good measure:
meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
with open(meta_path, "w") as jf:
    json.dump(data_dict, jf)
