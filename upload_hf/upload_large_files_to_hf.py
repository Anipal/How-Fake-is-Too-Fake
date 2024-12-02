"""
Upload fulldataset files to huggingface: `semi-truths/Semi-Truths/inpainting/CityScapes`.
"""

import os
import pdb
import tarfile
import shutil
from glob import glob
from tqdm import tqdm
from huggingface_hub import HfApi

api = HfApi()

# api.create_repo(
#     "semi-truths/Semi-Truths-Evalset-2",
#     token="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",
#     repo_type="dataset",
# )

root_dir = "/data/jkruk3/half-truths/mistral_inpainting/"

data_dirs = [
    # "/data/jkruk3/half-truths/mistral_inpainting/CelebAHQ",
    "/data/jkruk3/half-truths/mistral_inpainting/ADE20K",
]
comp_dir = "/data/jkruk3/half-truths/data_samples/compressed_large_hf/"
if not os.path.exists(comp_dir):
    os.mkdir(comp_dir)

subd = [
    "Kandinsky_2_2",
    "StableDiffusion_v4",
    "StableDiffusion_XL",
    "StableDiffusion_v5",
    "OpenJourney",
]

# inpainting/CityScapes

for data_dir in data_dirs:
    for root, dirs, files in os.walk(data_dir):
        for dir in dirs:
            if dir in subd:

                sub_dir = root.replace(root_dir, "")
                dest_data_path = os.path.join(comp_dir, sub_dir)
                if not os.path.exists(dest_data_path):
                    os.mkdir(dest_data_path)

                dest_path = os.path.join(comp_dir, sub_dir, dir)
                if not os.path.exists(dest_path):
                    os.mkdir(dest_path)

                data_path = os.path.join(root, dir)
                model_files = glob(data_path + "/*")
                num_files = len(model_files)

                for i in range(3):
                    tar_name = f"{sub_dir}_{dir}_{i}.tar.bz2"
                    if not os.path.exists(os.path.join(dest_path, tar_name)):
                        model_subset = model_files[
                            i * int(num_files / 3) : (i + 1) * int(num_files / 3)
                        ]
                        with tarfile.open(tar_name, "w:bz2") as tar:
                            for i in tqdm(range(len(model_subset))):
                                tar.add(
                                    model_subset[i],
                                    arcname=os.path.basename(model_subset[i]),
                                )
                        shutil.move(tar_name, dest_path)
                        print(f"Created tar file: {tar_name}")


for root, dirs, files in os.walk(comp_dir):
    for filename in files:
        parent_dir = root.split("/")[-1]
        dataset = root.split("/")[-2]
        if (parent_dir in subd) and (dataset in root):
            path_ = os.path.join(root, filename)
            relative_path = os.path.join("inpainting", path_.replace(comp_dir, ""))
            print(relative_path, path_)

            try:
                api.delete_file(
                    path_in_repo=relative_path,
                    repo_id="semi-truths/Semi-Truths",
                    repo_type="dataset",
                    token="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",
                )
            except:
                print("File already deleted")

            api.upload_file(
                path_or_fileobj=path_,
                path_in_repo=relative_path,
                repo_id="semi-truths/Semi-Truths",
                repo_type="dataset",
                token="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",
            )


# #uplaoding metadata files:
for data_dir in data_dirs:
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if ".csv" in filename and any([x in filename for x in subd]):

                ds_dir = data_dir.split("/")[-1].replace("/", "")
                sub_ds_dir = filename.replace("_meta.csv", "").replace(ds_dir + "_", "")

                dpath = os.path.join(root, filename)
                relpath = os.path.join("inpainting", ds_dir, sub_ds_dir, filename)

                try:
                    api.delete_file(
                        path_in_repo=os.path.join(
                            "inpainting", ds_dir, sub_ds_dir, filename
                        ),
                        repo_id="semi-truths/Semi-Truths",
                        repo_type="dataset",
                        token="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",
                    )
                except:
                    print("File already deleted")

                api.upload_file(
                    path_or_fileobj=os.path.join(root, filename),
                    path_in_repo=os.path.join(
                        "inpainting", ds_dir, sub_ds_dir, filename
                    ),
                    repo_id="semi-truths/Semi-Truths",
                    repo_type="dataset",
                    token="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",
                )

# del_paths = [
#     "CelebAHQ_OpenJourney_0.tar.bz2",
#     "CelebAHQ_StableDiffusion_v5_0.tar.bz2",
# ]

# for file in del_paths:
#     api.delete_file(
#         path_in_repo=file,
#         repo_id="semi-truths/Semi-Truths",
#         repo_type="dataset",
#         token="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",
#     )

# del_folders = [
#     "inpainting/ADE20K",
#     "inpainting/CelebAHQ",
#     "inpainting/mistral_inpainting",
# ]

# for file in del_folders:
#     api.delete_folder(
#         path_in_repo=file,
#         repo_id="semi-truths/Semi-Truths-Evalset",
#         repo_type="dataset",
#         token="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",
#     )
