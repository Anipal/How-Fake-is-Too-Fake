"""
Helper functions that can be used across datasets
for processes before image perturbation scheme.
"""

import os
import numpy as np
import pandas as pd
import cv2
# from datasets import load_dataset
from tqdm import tqdm
import pdb

# import smartcrop
from PIL import Image


def mask_sa_ratio(mask):
    h, w = mask.size
    # mask_ = mask.convert("1")
    mask_ = np.array(mask)
    mask_ = cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY)
    img_arr = np.asarray(mask_)

    img_sa = h * w
    mask_sa = np.count_nonzero(img_arr)
    ratio = mask_sa / img_sa
    print(f"mask ratio -- {ratio}")
    return ratio


def crop_and_resize(image, main_mask, cropper, dest_size=256):
    """
    This function takes an image and mask of the
    Human Parsing Dataset, smart-crops both, and
    resizes them to the dest_size.

    Input:
        - image (PIL)
        - main_mask (PIL)
        - cropper (smartcrop object)
        - dest_size (int)

    Output:
        - image (PIL)
        - main_mask (PIL)
    """

    # determine smaller image side size:
    h, w = image.size
    if h > w:
        s = w
    else:
        s = h

    result = cropper.crop(image, s, s)
    result_flat = []
    for c in result["crops"]:
        score = c.pop("score")
        c["detail"] = score["detail"]
        c["saturation"] = score["saturation"]
        c["skin"] = score["skin"]
        c["total"] = score["total"]
        result_flat.append(c)

    crop_df = pd.DataFrame(result_flat)
    best_detail = crop_df.loc[crop_df["detail"].idxmax()]

    image_detail = image.crop((best_detail["x"].item(), best_detail["y"].item(), s, s))
    image_detail = image_detail.resize((dest_size, dest_size))
    main_mask_detail = main_mask.crop(
        (best_detail["x"].item(), best_detail["y"].item(), s, s)
    )
    main_mask_detail = main_mask_detail.resize((dest_size, dest_size))

    return image_detail, main_mask_detail


def verify_mask(mask, threshold=0.005):
    """
    Verify that mask has not been removed during
    smartcrop and rescale. HPD masks are not "clean",
    so some pixels remain outside of mask. Mask
    needs to have SA ratio to image of > 5%.
    """

    h, w = mask.size
    mask_ = cv2.cvtColor(np.array(mask), cv2.COLOR_BGR2GRAY)
    # mask_ = mask.convert("1")
    img_arr = np.asarray(mask_)

    img_sa = h * w
    mask_sa = np.count_nonzero(img_arr)  # white color --> 225
    ratio = mask_sa / img_sa

    return ratio, ratio > threshold


def flatten_mask(main_mask, mask_dict):
    """
    Seperates the cropped and scaled main_mask into
    individual "keep" masks. Returns a dictionary of
    mask names as keys, and mask PIL objects as values.
    """

    h, w = main_mask.size
    img_arr = np.asarray(main_mask)
    new_img_arr = np.zeros((w, h))
    masks = {}

    # converting mask to keep masks:
    for key, value in mask_dict.items():
        if int(key) in img_arr:
            for i in range(w):
                for j in range(h):
                    try:
                        if img_arr[i, j] == int(key):
                            new_img_arr[i, j] = 0
                        else:
                            new_img_arr[i, j] = 255
                    except:
                        pdb.set_trace()

            new_mask = Image.fromarray(new_img_arr)
            if new_mask.mode != "RGB":
                new_mask = new_mask.convert("RGB")

            # filter mask by minimum size:
            ratio, rb = verify_mask(new_mask)
            if rb:
                masks[value] = (new_mask, ratio)

    return masks


def center_crop(im, new_size):
    width, height = im.size  # Get dimensions

    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


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
