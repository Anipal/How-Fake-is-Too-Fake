import argparse
from magnitude_change import *
import pdb
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(description="Color Scheme for Skyscapes and Uavid")parser.add_argument(
    "--root_csv",
    type=str,
    default="/srv/share4/apal72/half-truths/csvs/consolidated_meta_no_helen.csv",
    help="main csv",
)
parser.add_argument(
    "--root_dir",
    type=str,
    default="/srv/share4/apal72/half-truths/data/Half_Truths_Dataset",
    help="root directory to the images",
)
parser.add_argument(
    "--model", type=str, default="mistral_inpainting", help="model name"
)
parser.add_argument("--save_dir", type=str, default=None)

args = parser.parse_args()

datasets = []
models = []

data = pd.read_csv(args.root_csv)
data = data.sort_values(by="method", ascending=True)
# pdb.set_trace()

try:
    real_paths = list(
        args.root_dir + "/" + "images" + "/" + data["dataset"] + "/" + data["img_id"]
    )
    fake_paths = [
        os.path.join(args.root_dir, i.split("Half_Truths_Dataset/")[-1])
        for i in data["perturbed_path"]
    ]
    perturbed_label = data["perturbed_label"]

except:
    real_paths = list(args.root_dir + "/" + data["image_path"])
    fake_paths = list(args.root_dir + "/" + data["perturbed_image_path"])
    perturbed_label = data["perturbed_mask_name"]


og_label = data["mask_name"]
perturb_type = data["method"][0]
change_calc = Change(save_dir=args.save_dir)

metrics = []
# real_paths = real_paths[:100]
for i in tqdm(range(len(real_paths)), total=len(real_paths)):
    if data.iloc[i].method != perturb_type:
        perturb_type = data.iloc[i].method
        change_calc = Change(perturb_type=perturb_type)

    try:
        metrics.append(
            change_calc.calc_metrics(
                real_paths[i], fake_paths[i], og_label[i], perturbed_label[i]
            )
        )
    except Exception as error:
        print(f"ERROR:   {error}")
        pdb.set_trace()

columns = [
    "dreamsim",
    "lpips_score",
    "sen_sim",
    "clip_sim",
    "mse_rgb",
    "mse_gray",
    "ssim_rgb",
    "ssim_gray",
    "ratio_rgb",
    "ratio_gray",
    "largest_component_size_gray",
    "largest_component_size_rgb",
    "cc_cluters_rgb",
    "cc_clusters_gray",
    "cluster_dist_rgb",
    "cluster_dist_gray",
]
metrics_data = pd.DataFrame(metrics, columns=columns)
final_data = pd.concat([data, metrics_data], axis=1)

name = os.path.splitext(args.root_csv.split("/")[-1])[0] + "_mag-metrics.csv"
pdb.set_trace()
if not (os.path.exists(os.path.join(args.root_dir, "mag_csvs"))):
    os.makedirs(os.path.join(args.root_dir, "mag_csvs"))
final_data.to_csv(os.path.join(args.root_dir, "mag_csvs", name), index=False)
name = os.path.splitext(args.root_csv.split('/')[-1])[0] + 'trial_mag-metrics.csv'
# pdb.set_trace()
if not(os.path.exists(os.path.join(args.root_dir,'mag_csvs'))):
    os.makedirs(os.path.join(args.root_dir,'mag_csvs'))
final_data.to_csv(os.path.join(args.root_dir,'mag_csvs',name), index=False)





