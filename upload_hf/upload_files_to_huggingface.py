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


data_dir = "/data/jkruk3/half-truths/data_samples/eval_sample_5p/full_hf/"
comp_dir = "/data/jkruk3/half-truths/data_samples/eval_sample_5p/compressed_full_hf/"
if not os.path.exists(os.path.join(comp_dir, "original")):
    os.mkdir(os.path.join(comp_dir, "original"))

# for root, dirs, files in os.walk(data_dir):
#     subd = [
#         "ADE20K",
#         "CelebAHQ",
#         "CityScapes",
#         "HumanParsing",
#         "OpenImages",
#         "SUN_RGBD",
#     ]
#     for dir in dirs:
#         if dir in subd:

#             sub_dir = root.replace(data_dir, "")
#             if not os.path.exists(os.path.join(comp_dir, sub_dir)):
#                 os.mkdir(os.path.join(comp_dir, sub_dir))

#             tar_name = dir + ".tar.bz2"
#             dest_path = os.path.join(comp_dir, sub_dir)
#             if not os.path.exists(tar_name) and not os.path.exists(
#                 os.path.join(dest_path, tar_name)
#             ):
#                 data_path = os.path.join(root, dir)
#                 files_ = glob(data_path + "/*")
#                 with tarfile.open(tar_name, "w:bz2") as tar:
#                     for i in tqdm(range(len(files_))):
#                         tar.add(files_[i], arcname=os.path.basename(files_[i]))

#                 shutil.move(tar_name, dest_path)

#         elif "p2p_masks" in root:
#             if not os.path.exists(os.path.join(comp_dir, "p2p_masks")):
#                 os.mkdir(os.path.join(comp_dir, "p2p_masks"))

#             if dir in ["OpenJourney", "StableDiffusion_v4", "StableDiffusion_v5"]:

#                 sub_dir = root.replace(data_dir, "")
#             if not os.path.exists(os.path.join(comp_dir, sub_dir)):
#                 os.mkdir(os.path.join(comp_dir, sub_dir))

#             tar_name = dir + ".tar.bz2"
#             if not os.path.exists(tar_name) and not os.path.exists(
#                 os.path.join(dest_path, tar_name)
#             ):
#                 data_path = os.path.join(root, dir)
#                 files_ = glob(data_path + "/*")
#                 with tarfile.open(tar_name, "w:bz2") as tar:
#                     for i in tqdm(range(len(files_))):
#                         tar.add(files_[i], arcname=os.path.basename(files_[i]))

#                 dest_path = os.path.join(comp_dir, sub_dir)
#                 shutil.move(tar_name, dest_path)

# for filename in files:
#     path_ = os.path.join(root, filename)
#     relative_path = path_.replace(data_dir, "")
#     # print(relative_path, path_)
#     api.upload_file(
#         path_or_fileobj=path_,
#         path_in_repo=relative_path,
#         repo_id="semi-truths/Semi-Truths-Evalset",
#         repo_type="dataset",
#         token="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",
#     )

# api.upload_file(
#     path_or_fileobj="/data/jkruk3/half-truths/data_samples/eval_sample_5p/full_hf/metadata.csv",
#     path_in_repo="metadata.csv",
#     repo_id="semi-truths/Semi-Truths-Evalset",
#     repo_type="dataset",
#     token="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",
# )

# for root, dirs, files in os.walk(comp_dir):
#     for file in files:
#         global_path = os.path.join(root, file)
#         rel_path = global_path.replace(comp_dir, "")
#         api.upload_file(
#             path_or_fileobj=global_path,
#             path_in_repo=rel_path,
#             repo_id="semi-truths/Semi-Truths-Evalset",
#             repo_type="dataset",
#             token="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",
#         )

del_paths = [
    "semi_truths_10p_0.tar.bz2",
    "semi_truths_10p_1.tar.bz2",
    "semi_truths_10p_2.tar.bz2",
]

for file in del_paths:
    api.delete_folder(
        path_in_repo=file,
        repo_id="semi-truths/Semi-Truths-Evalset",
        repo_type="dataset",
        token="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",
    )


# api.upload_folder(
#     folder_path="/data/jkruk3/half-truths/data_samples/eval_sample_5p/compressed_full_hf/",
#     repo_id="semi-truths/Semi-Truths-Evalset",
#     token="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",
#     repo_type="dataset",
# )

# api.upload_folder(

#     folder_path="/raid/mphute6/HalfTruths/inpainting_data/half-truths/half-truths/inpainting/inpainting_data",

#     repo_id="semi-truths/Semi-Truths",

#     repo_type="dataset",

# )


# api.delete_file(

#     path_in_repo = "CityScapes",

#     repo_id="semi-truths/Semi-Truths",

#     repo_type="dataset",

# )
