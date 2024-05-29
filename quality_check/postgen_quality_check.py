import torch
import os
from PIL import Image
from utils import *
from tqdm import tqdm
import ipdb
from tqdm import tqdm
import csv
import pandas as pd

torch.set_default_tensor_type("torch.cuda.FloatTensor")

CSV_READ_FILE = "/data/jkruk3/half-truths/data_samples/eval_sample_5p/prompt-prompt_subset_qual_1/prompt-prompt-llama.csv"

# inpainting
# column_map = {
#     "image_id": 0,
#     "image_path": 1,
#     "dataset": 2,
#     "entities": 3,
#     "mask_id": 4,
#     "mask_path": 5,
#     "mask_name": 6,
#     "ratio": 7,
#     "perturbed_image_id": 8,
#     "method": 9,
#     "model": 10,
#     "perturbed_mask_name": 11,
#     "perturbed_image_path": 12,
#     "classes": 13,
#     "sem_magnitude": 14,
#     "quality_flag": 15,
#     "class": 16,
#     "label": 17,
#     "language_model": 18,
# }

# p2p
column_map = {
    "image_id": 0,
    "image_path": 1,
    "dataset": 2,
    "entities": 3,
    # "mask_id": 4,
    # "mask_path": 5,
    "mask_name": 4,
    # "ratio": 7,
    "perturbed_image_id": 5,
    "method": 6,
    "model": 7,
    "perturbed_mask_name": 8,
    "perturbed_image_path": 9,
    "classes": 10,
    # "class": 10,
    "label": 11,
    "language_model": 12,
    "sem_magnitude": 13,
    # "quality_flag": 14,
}

PATH_TO_DATA_PARENT = "/data/jkruk3"
PATH_TO_PERT_PARENT = (
    "/data/jkruk3/half-truths/data_samples/eval_sample_5p/prompt-prompt_subset_qual_1"
)
CSV_POSTGEN_QC = "/data/jkruk3/half-truths/data_samples/eval_sample_5p/prompt-prompt_subset_qual_1/postgen_quality_check.csv"

# ordering csv columns to match map:
df_ = pd.read_csv(CSV_READ_FILE)
if "img_id" in df_.columns:
    df_ = df_.rename(
        columns={
            "img_id": "image_id",
            "perturbed_label": "perturbed_mask_name",
            "perturbed_path": "perturbed_image_path",
        }
    )
df_ = df_[list(column_map.keys())]
df_.to_csv(CSV_READ_FILE, index=False)


def main():

    f = open(CSV_READ_FILE, "r")
    file = csv.reader(f)
    header = next(file)

    header_qc = header
    header_qc.extend(["cap2_img2", "direct_sim"])

    out_f = open(CSV_POSTGEN_QC, "w")
    writer = csv.writer(out_f)
    writer.writerow(header_qc)

    for row in tqdm(file):
        if len(row[column_map["image_path"]]):
            method = row[column_map["method"]]
            model = row[column_map["model"]]

            orig_img_path = os.path.join(
                PATH_TO_DATA_PARENT, row[column_map["image_path"]]
            )
            if ".png" not in orig_img_path and ".jpg" not in orig_img_path:
                try:
                    orig_img_path_ = orig_img_path + ".png"
                    orig_img = Image.open(orig_img_path_)

                except FileNotFoundError:
                    orig_img_path_ = orig_img_path + ".jpg"
                    orig_img = Image.open(orig_img_path_)
            else:
                orig_img = Image.open(orig_img_path)

            orig_caption = row[column_map["mask_name"]]
            perturbed_img_path = os.path.join(
                PATH_TO_PERT_PARENT, row[column_map["perturbed_image_path"]]
            )
            perturbed_img = Image.open(perturbed_img_path)

            ### DEBUG LINE, LEAVE COMMENTED
            # perturbed_img = orig_img

            perturbed_caption = row[column_map["perturbed_mask_name"]]

            cap2_img2 = calculate_image_caption_clip_similarity(
                perturbed_img, perturbed_caption
            )
            direct_sim = calculate_directional_similarity(
                orig_img, orig_caption, perturbed_img, perturbed_caption
            )

            row.extend([cap2_img2, direct_sim])

            writer.writerow(row)

    f.close()
    out_f.close()


if __name__ == "__main__":
    main()
