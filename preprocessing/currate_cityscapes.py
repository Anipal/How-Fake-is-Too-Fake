"""
This script processes the CityScapes dataset for
the purposes of creating a perturbation dataset.
"""

import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import ipdb
from PIL import Image, ImageDraw, ImageOps
from collections import namedtuple
import json
import matplotlib.pyplot as plt
from utils import verify_mask, flatten_mask, center_crop, transform_mask
import geopandas as gpd
from shapely.geometry import Polygon, mapping
from shapely import affinity


DATA_DIR = "/data/jkruk3/CityScapes_gtFine/train"
ORG_IMG_DIR = "/data/jkruk3/CityScapes_leftImg8bit/train"

OUTPUT_DIR = "/data/jkruk3/CityScapes_256"
IMG_DIR = "/data/jkruk3/CityScapes_256/images"
MASK_DIR = "/data/jkruk3/CityScapes_256/masks"
dest_size = 256

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
if not os.path.exists(IMG_DIR):
    os.mkdir(IMG_DIR)
if not os.path.exists(MASK_DIR):
    os.mkdir(MASK_DIR)


# a label and all meta information
Label = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class
        "id",  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images
        # An ID of -1 means that this label does not have an ID and thus
        # is ignored when creating ground truth images (e.g. license plate).
        # Do not modify these IDs, since exactly these IDs are expected by the
        # evaluation server.
        "trainId",  # Feel free to modify these IDs as suitable for your method. Then create
        # ground truth images with train IDs, using the tools provided in the
        # 'preparation' folder. However, make sure to validate or submit results
        # to our evaluation server using the regular IDs above!
        # For trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the inverse
        # mapping, we use the label that is defined first in the list below.
        # For example, mapping all void-type classes to the same ID in training,
        # might make sense for some approaches.
        # Max value is 255!
        "category",  # The name of the category that this label belongs to
        "categoryId",  # The ID of this category. Used to create ground truth images
        # on category level.
        "hasInstances",  # Whether this label distinguishes between single instances or not
        "ignoreInEval",  # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not
        "color",  # The color of this label
    ],
)


labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
    Label("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
    Label("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
    Label("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
    Label("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
    Label("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
    Label("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
    Label("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
    Label("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
    Label("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
    Label("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
    Label("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
    Label("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
    Label("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
    Label("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
    Label("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
    Label("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
    Label("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
    Label("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
    Label("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
    Label("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
    Label("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
    Label("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
    Label("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
    Label("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
    Label("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
    Label("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
    Label("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
    Label("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
    Label("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
    Label("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
    Label("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
    Label("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
    Label("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
    Label("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
]


def compute_mask_area_ratio(coords, im_area):
    """
    Takes an object dictionary, returns
    the area of the object masks.
    """
    try:
        polygon = Polygon(coords)
        poly_df = gpd.GeoDataFrame(geometry=[polygon])
        area = poly_df.area.item()
        area_ratio = area / im_area
    except:
        area_ratio = None
    return area_ratio


def contains_people(img, objects, person_thresh=0.01):
    w, h = img.size
    img_sa = w * h

    person_objs = [x for x in objects if x["label"] == "person"]
    if len(person_objs) > 0:
        ratio_passes = []
        for obj in person_objs:
            polygon = obj["polygon"]
            polygon.append(polygon[0])
            mask_ratio = compute_mask_area_ratio(polygon, img_sa)
            pass_ = mask_ratio >= person_thresh
            ratio_passes.append(pass_)
        return any(ratio_passes)
    else:
        return False


# _______________________________________________________________ #


img_ids = []
meta_files = []
mask_image_files = []
org_image_files = []
ppl_count = 0

for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if "_gtFine_polygons.json" in file:
            meta_files.append(os.path.join(root, file))
            id_ = file.replace("_gtFine_polygons.json", "")
            img_ids.append(id_)
        elif "_gtFine_labelIds.png" in file:
            mask_image_files.append(os.path.join(root, file))
            id_ = file.replace("_gtFine_labelIds.png", "")
            img_ids.append(id_)

for root, dirs, files in os.walk(ORG_IMG_DIR):
    for file in files:
        if ".png" in file:
            org_image_files.append(os.path.join(root, file))


# checking if intermediate metadata has been saved:
out_file = os.path.join(OUTPUT_DIR, "metadata.json")
if os.path.exists(out_file):
    with open(out_file, "r") as jf:
        data_dict = json.load(jf)
else:
    data_dict = {}


for id in tqdm(img_ids, total=len(img_ids)):
    # checking if image has already been processed:
    dest_img_path_ = os.path.join(IMG_DIR, f"{id}.png")
    if not os.path.exists(dest_img_path_):
        meta_f = [x for x in meta_files if id in x][0]
        m_img_f = [x for x in mask_image_files if id in x][0]
        org_img_f = [x for x in org_image_files if id in x][0]
        with open(meta_f, "r") as jf:
            meta = json.load(jf)

        m_img = Image.open(m_img_f)
        w, h = m_img.size

        # get new image size dimentions
        llen = h if h < w else w
        scale_ratio = dest_size / llen
        if h < w:
            h_new = dest_size
            w_new = scale_ratio * w
        else:
            h_new = scale_ratio * w
            w_new = dest_size

        objects = []

        # person_objs = [x for x in meta["objects"] if x["label"] == "person"]
        # if len(person_objs) > 0:
        if contains_people(m_img, meta["objects"]):
            ppl_count = ppl_count + 1

            for j, obj in enumerate(meta["objects"]):
                polygon = obj["polygon"]
                polygon.append(
                    polygon[0]
                )  # repeat the first point to create a 'closed loop'

                xs, ys = zip(*polygon)  # create lists of x and y values
                poly2 = []
                for i in range(len(xs)):
                    z = (xs[i], ys[i])
                    poly2.append(z)

                img2 = m_img.copy()
                draw = ImageDraw.Draw(img2)
                draw.polygon(poly2, fill="#FFFFFF")

                mask_img = Image.blend(m_img, img2, 1)
                mask_img = transform_mask(mask_img)
                inverted_mask_img = ImageOps.invert(mask_img)

                inverted_mask_img = inverted_mask_img.resize((int(w_new), int(h_new)))
                inverted_mask_img = center_crop(inverted_mask_img, dest_size)
                ratio, threshold = verify_mask(inverted_mask_img)

                if threshold:
                    m_id = str(j).zfill(3)
                    m_path = os.path.join(MASK_DIR, f"{id}_instance{m_id}.png")
                    inverted_mask_img.save(m_path)

                    m_ = {
                        "id": f"{id}_instance{m_id}",
                        "mask_path": m_path,
                        "entity": obj["label"],
                        "ratio": ratio,
                    }
                    objects.append(m_)

            org_img = Image.open(org_img_f)
            org_img = org_img.resize((int(w_new), int(h_new)))
            org_img = center_crop(org_img, dest_size)
            org_dest = os.path.join(IMG_DIR, f"{id}.png")
            org_img.save(org_dest)
            # org_img.save(os.path.join("test_masks", f"{id}.png"))

            meta = {
                "src_image_path": org_img_f,
                "image_path": org_dest,
                "objects": objects,
            }
            data_dict[id] = meta

print(f"Number of images containing people:    {ppl_count}")
try:
    with open(out_file, "w") as jf:
        json.dump(data_dict, jf)
except:
    ipdb.set_trace()
