import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import ipdb
import smartcrop
from PIL import Image


DATA_DIR = "/data/jkruk3/HumanParsing_256"
IMG_DIR = "/data/jkruk3/HumanParsing_256/images"
MASK_DIR = "/data/jkruk3/HumanParsing_256/masks"

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(IMG_DIR):
    os.mkdir(IMG_DIR)
if not os.path.exists(MASK_DIR):
    os.mkdir(MASK_DIR)

# the pixel values of the mask pertain to specific
# semantic segmentations, as shown below.
mask_dict = {
    "0": "Background",
    "1": "Hat",
    "2": "Hair",
    "3": "Sunglasses",
    "4": "Upper-clothes",
    "5": "Skirt",
    "6": "Pants",
    "7": "Dress",
    "8": "Belt",
    "9": "Left-shoe",
    "10": "Right-shoe",
    "11": "Face",
    "12": "Left-leg",
    "13": "Right-leg",
    "14": "Left-arm",
    "15": "Right-arm",
    "16": "Bag",
    "17": "Scarf"
}


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
    for c in result['crops']:
        score = c.pop('score')
        c['detail'] = score['detail']
        c['saturation'] = score['saturation']
        c['skin'] = score['skin']
        c['total'] = score['total']
        result_flat.append(c)

    crop_df = pd.DataFrame(result_flat)
    best_detail = crop_df.loc[crop_df['detail'].idxmax()]

    image_detail = image.crop((best_detail['x'].item(), best_detail['y'].item(), s, s))
    image_detail = image_detail.resize((dest_size, dest_size))
    main_mask_detail = main_mask.crop((best_detail['x'].item(), best_detail['y'].item(), s, s))
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
    mask_ = mask.convert('1')
    img_arr = np.asarray(mask_)

    img_sa = h * w
    mask_sa = abs(img_sa - np.count_nonzero(img_arr))
    ratio = mask_sa/img_sa

    # ipdb.set_trace()
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
                        if img_arr[i,j] == int(key):
                            new_img_arr[i,j] = 0
                        else: 
                            new_img_arr[i,j] = 255
                    except:
                        ipdb.set_trace()

            new_mask = Image.fromarray(new_img_arr)
            if new_mask.mode != 'RGB':
                new_mask = new_mask.convert('RGB')

            # filter mask by minimum size: 
            ratio, rb = verify_mask(new_mask)
            if rb:
                masks[value] = (new_mask, ratio)

    return masks


def save_image_data(img_id, image, masks):
    """
    This function saves images and masks in the
    appropriate locations, then returns the structured
    metadata for the image, to be joined in a dataset
    DataFrame further downstream.
    """

    img_path = os.path.join(IMG_DIR, img_id + ".png")
    meta = {
        "img_id" : img_id,
        "img_path" : img_path,
        "elements" : list(mask_dict.keys()),
        "masks" : [],
        "dataset" : "Human Parsing Dataset"
    }

    image.save(img_path)
    for key, (mask, ratio) in masks.items():
        mask_path = os.path.join(MASK_DIR, f"{img_id}_{key.replace('-', '')}.png")
        mask.save(mask_path)

        m_ = {
            "element" : key,
            "mask_path" : mask_path,
            "area_ration" : ratio
        }
        meta["masks"].append(m_)
    return meta


def process_HPD(data, cropper):
    """
    Main function for processing and saving 
    the Human Parsing Dataset.
    """

    meta = pd.DataFrame()

    for i in tqdm(range(len(data['train']))):
        image = data['train'][i]['image']
        mask = data['train'][i]['mask']

        h, w = image.size
        if h > w: 
            ratio = w/h 
        else:
            ratio = h/w

        # filtering out images that are too long
        if ratio > 0.65:
            img_id = f"img_{str(i).zfill(5)}"
            image_sm, mask_sm = crop_and_resize(image, mask, cropper, dest_size=256)
            masks = flatten_mask(mask_sm, mask_dict)
            img_meta = save_image_data(img_id, image_sm, masks)
            meta = pd.concat([meta, pd.DataFrame(img_meta)])

if __name__ == "__main__":

    # loading HPD from huggingface:
    data = load_dataset("mattmdjaga/human_parsing_dataset")

    # loading smart cropper object
    # see: https://github.com/smartcrop/smartcrop.py 
    cropper = smartcrop.SmartCrop()

    # processing & saving HPD:
    process_HPD(data, cropper)

        

