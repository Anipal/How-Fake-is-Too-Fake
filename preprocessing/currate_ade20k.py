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
import cv2

"""
Currate the ADE20K dataset to only contain image, mask, and meta data of
people in natural scenes. This tasks requires the following:
    -   Identify images with people
    -   Remove any person info if the mask is less than 10% of the image.
        - This means masks and meta data
    - Filter out any images with no people remaining
    - Organize & compress
"""

# important data directories
DATA_DIR = "/data/jkruk3/"
ADE_DIR = "/data/jkruk3/ADE20K_2021_17_01/images/ADE"
FINAL_DATA_DIR = "/data/jkruk3/ADE20K_People"
FINAL_IMG_DIR = "/data/jkruk3/ADE20K_People/images"

PERSON_SIZE_THRESH = 0.1


def create_path_dir(ADE_DIR):
    """
    Create dictionary with essential 
    img-relevant data paths.
    """
    print("Creating data path dictionary ...")
    data_path_dic = {}
    for root, dirs, files in os.walk(ADE_DIR):

        if len(dirs) > 0 and "ADE_train_" in dirs[0]:
            for dr in dirs:
                img_dir = {}

                # image paths
                img_f = [x for x in files if dr in x and ".jpg" in x][0]
                img_dir["image"] = os.path.join(root, img_f)

                # full segmentation image path
                seg_f = [x for x in files if dr in x and "seg.png" in x][0]
                img_dir["segmentation"] = os.path.join(root, seg_f)

                # object mask paths
                masks = []
                for m in os.listdir(os.path.join(root, dr)):
                    m_f = os.path.join(root, dr, m)
                    masks.append(m_f)
                img_dir["object_masks"] = masks

                # metadata path
                meta_f = [x for x in files if dr in x and ".json" in x][0]
                img_dir["raw_meta_path"] = os.path.join(root, meta_f)

                data_path_dic[dr] = img_dir

    return data_path_dic


def compute_mask_area_ratio(obj, im_area):
    """
    Takes an object dictionary, returns
    the area of the object masks.
    """
    coords = list(zip(obj["polygon"]["x"], obj["polygon"]["y"]))
    try:
        polygon = Polygon(coords)
        poly_df = gpd.GeoDataFrame(geometry=[polygon])

        area = poly_df.area.item()
        area_ratio = area / im_area
    except:
        area_ratio = None
    return area_ratio


def transform_mask(mask_img):
    """
    Transforms masks provided in the ADE20k dataset
    into true binary masks. (Some of them has grey regions
    for some reason - RePaint wont take it).

    mask_img must be opened with PIL.
    """
    pixel_values = mask_img.load()
    for x in range(mask_img.width):
        for y in range(mask_img.height):
            if pixel_values[x, y] < 255:
                pixel_values[x, y] = 0

    return mask_img


def filter_dataset(data_path_dic, PERSON_SIZE_THRESH):
    """
    Takes an image data dictionary, and returns
    a filtered version that only contains images
    with people that are larger than the provided
    image ratio threshold.
    """
    print("Filtering dataset on people ...")
    for img in tqdm(data_path_dic.keys(), total=len(data_path_dic.keys())):
        json_path = data_path_dic[img]["raw_meta_path"]
        with open(json_path, "r") as jf:
            meta = json.load(jf)

        meta = meta["annotation"]
        del meta["folder"]
        del meta["filename"]

        # computing image area (for later steps):
        img_area = meta["imsize"][0] * meta["imsize"][1]

        p_objs = []
        for obj in meta["object"]:
            hyps1 = [x.split(", ") for x in obj["hypernym"]]
            hyps = [item for sublist_ in hyps1 for item in sublist_]
            if "person" in hyps:
                p_objs.append(obj)

        if len(p_objs) > 0:
            # verifying that person is large enough
            # i.e. passes surface area threshold
            thresh_passes = []

            for p in p_objs:
                ratio = compute_mask_area_ratio(p, img_area)
                thresh_pass = ratio >= PERSON_SIZE_THRESH
                thresh_passes.append(thresh_pass)

            if any(thresh_passes):
                # make sure to remove people that are too small
                p_idxs = [i for i, x in enumerate(thresh_passes) if not x]
                small_p_obs = [p_objs[i] for i in p_idxs]

                elements = []
                for obj in meta["object"]:
                    if obj not in small_p_obs:
                        obj["name"] = obj["raw_name"]
                        del obj["name_ndx"]
                        del obj["raw_name"]
                        del obj["attributes"]
                        del obj["occluded"]
                        del obj["crop"]
                        del obj["parts"]

                        elements.append(obj["name"])
                        obj["area_ratio"] = compute_mask_area_ratio(obj, img_area)
                        try:
                            del obj["save_date"]
                        except:
                            continue

                data_path_dic[img]["elements"] = list(set(elements))
                data_path_dic[img]["metadata"] = meta

    data_path_dic_people = {}
    for key, value in data_path_dic.items():
        if "metadata" in value.keys():
            data_path_dic_people[key] = value

    print(f"Length of original data:   {len(data_path_dic)}")
    print(f"Length of people-centric data:   {len(data_path_dic_people)}")
    return data_path_dic_people


def filter_images(data_path_dic_people, FINAL_IMG_DIR):
    """
    Transfers images & mask data of the images that 
    passed data filtration to a desired location.
    """
    print("Transfering final images ...")
    if not os.path.exists(FINAL_IMG_DIR):
        os.mkdir(FINAL_IMG_DIR)

    for img in tqdm(data_path_dic_people.keys(), total=len(data_path_dic_people.keys())):
        img_data = data_path_dic_people[img]
        shutil.copy(img_data["image"], os.path.join(FINAL_IMG_DIR, img + ".jpg"))
        shutil.copy(img_data["segmentation"], os.path.join(FINAL_IMG_DIR, img + ".png"))

        try:
            shutil.copytree(img_data["image"].replace(".jpg", ""), os.path.join(FINAL_IMG_DIR, img))
        except:
            shutil.rmtree(os.path.join(FINAL_IMG_DIR, img))
            shutil.copytree(img_data["image"].replace(".jpg", ""), os.path.join(FINAL_IMG_DIR, img))

        img_data["image"] = os.path.join(FINAL_IMG_DIR, img + ".jpg")
        img_data["segmentation"] = os.path.join(FINAL_IMG_DIR, img + ".png")

    return data_path_dic_people


def scale_data(data_path_dic_people, IMG_DIR, IMG_OUT_DIR, size=256):
    """
    Takes a metadata dictionary, Image & mask directory,
    destination Image & mask directory, and desired size
    of shortest image edge. Scales all images, masks, and
    polygon metadata to desired size, then saves in
    destination directory. Returns new metadata dict. 
    """

    for img, img_data in tqdm(
        data_path_dic_people.items(), total=len(data_path_dic_people.items())
    ):
        h, w = img_data["metadata"]["imsize"][0], img_data["metadata"]["imsize"][1]
        llen = h if h < w else w
        scale_ratio = size / llen

        # get new image size dimentions
        if h < w:
            h_new = size
            w_new = scale_ratio * w
        else:
            h_new = scale_ratio * w
            w_new = size
        img_area = h_new * w_new

        # scaling polygon metadata that describes object masks:
        rmv_objs = []
        for obj in img_data["metadata"]["object"]:
            try:
                coords = list(zip(obj["polygon"]["x"], obj["polygon"]["y"]))
                polygon = Polygon(coords)
                scaled_polygon = affinity.scale(polygon, xfact=scale_ratio, yfact=scale_ratio)
                poly_mapped = mapping(scaled_polygon)
                poly_coordinates = poly_mapped["coordinates"][0]

                x_ = []
                y_ = []
                for (x, y) in poly_coordinates:
                    x_.append(x)
                    y_.append(y)

                obj["polygon"]["x"] = x_
                obj["polygon"]["y"] = y_
                obj["area_ratio"] = compute_mask_area_ratio(obj, img_area)
            except:
                rmv_objs.append(obj)

        # remove and objects and masks that could not be scaled:
        img_data["metadata"]["object"] = [
            x for x in img_data["metadata"]["object"] if x not in rmv_objs
        ]

        # scaling and saving images:
        image = Image.open(img_data["image"])
        image = image.resize((int(w_new), int(h_new)))
        new_img_path = os.path.join(IMG_OUT_DIR, img + ".jpg")

        # scaling and saving masks:
        old_mask_path = img_data["image"].replace(".jpg", "")
        dest_mask_path = os.path.join(IMG_OUT_DIR, img)
        if not os.path.exists(dest_mask_path):
            os.mkdir(dest_mask_path)

        masks = [x for x in os.listdir(old_mask_path)]
        for m in masks:
            mask_img = Image.open(os.path.join(old_mask_path, m))
            mask_img = mask_img.resize((int(w_new), int(h_new)))
            mask_img = transform_mask(mask_img)
            inverted_mask_img = PIL.ImageOps.invert(mask_img)
            inverted_mask_img.save(os.path.join(dest_mask_path, m))

        image.save(new_img_path)
        img_data["image"] = os.path.join(IMG_OUT_DIR, img + ".jpg")

    return data_path_dic_people


if __name__ == "__main__":

    data_path_dic = create_path_dir(ADE_DIR)
    data_path_dic_people = filter_dataset(data_path_dic, PERSON_SIZE_THRESH)

    # transfering all images and masks in filtered dataset:
    filter_images(data_path_dic_people, FINAL_IMG_DIR)

    # saving to specified location:
    if not os.path.exists(FINAL_DATA_DIR):
        os.mkdir(FINAL_DATA_DIR)

    with open(os.path.join(FINAL_DATA_DIR, "metadata.json"), "w") as outfile:
        json.dump(data_path_dic_people, outfile)

    # UNCOMMENT IF: processed dataset needs to be scaled to a specific size
    size = 256
    SCALED_DIR = f"/data/jkruk3/ADE20K_People_{size}"
    SCALED_IMG_DIR = f"/data/jkruk3/ADE20K_People_{size}/images"

    # saving to specified location:
    if not os.path.exists(SCALED_DIR):
        os.mkdir(SCALED_DIR)
    if not os.path.exists(SCALED_IMG_DIR):
        os.mkdir(SCALED_IMG_DIR)

    data_path_dic_people_scaled = scale_data(
        data_path_dic_people, FINAL_IMG_DIR, SCALED_IMG_DIR, size=256
    )

    with open(os.path.join(SCALED_DIR, "metadata.json"), "w") as outfile:
        json.dump(data_path_dic_people_scaled, outfile)
