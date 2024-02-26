import argparse
import glob
import os

import torch
import csv
import pandas as pd
import torch.nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from DIRE_zp.utils.utils import get_network, str2bool, to_cuda
from CNNDetection.networks.resnet import resnet50

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-f", "--file", default="data/consolidated_meta_no_helen.csv", type=str, help="path to csv of image metadata"
)
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    default="CNNDetection/weights/blur_jpg_prob0.1.pth",
)
parser.add_argument("--use_cpu", action="store_true", help="uses gpu by default, turn on to use cpu")
parser.add_argument("--arch", type=str, default="resnet50")
parser.add_argument("--aug_norm", type=str2bool, default=True)
parser.add_argument("--model", type=str, default="dire", help="argsions: dire, cnnspot, defake")
parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('-j', '--workers', type=int, default=4, help='Number of workers')
parser.add_argument('--size_only', action='store_true', help='Only look at sizes of images in dataset')
parser.add_argument('--save_path', type=str, default="dire_preds.csv")

args = parser.parse_args()
pred_data = []
if os.path.isfile(args.file):
    print(f"Testing on file '{args.file}'")
    df = pd.read_csv(args.file)
    file_list = []
    file_list = df['image_path'].tolist()[:]
    file_list = ['./data/' + path for path in file_list]

trans = transforms.Compose(
    (
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    )
)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if args.model == "cnnspot":
    print("CNNSpot")
    model = resnet50(num_classes=1)

elif args.model == "dire":
    model = get_network(args.arch)

if args.model_path is not None:
    state_dict = torch.load(args.model_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
model.eval()
# if torch.cuda.is_available:
#     model.cuda()

for img_path in tqdm(file_list, dynamic_ncols=True, disable=len(file_list) <= 1):
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")
        img = trans(img)
        # if args.aug_norm:
        #     img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        in_tens = img.unsqueeze(0)
        if not args.use_cpu:
            in_tens = in_tens.cuda()

        with torch.no_grad():
            prob = model(in_tens).sigmoid().item()
        print(img_path, f"Prob of being synthetic: {prob:.4f}")
        pred_data.append((round(prob, 2), '1' if prob >= 0.50 else '0'))  # Append (rounded probability, prediction) tuple
    else:
        print("Cannot find file")
        pred_data.append(('NA', 'NA'))  # Append ('NA', 'NA') if file not found
        continue

print('predictions done, creating csv')
print(pred_data)
df['Prediction'] = [x[1] for x in pred_data]  # Extract predictions from (probability, prediction) tuple
df['Probability'] = [x[0] for x in pred_data]  # Extract probabilities
# print(df)
if not os.path.exists("preds"):
    os.makedirs("preds")

df.to_csv(os.path.join("preds", (args.model + "_preds.csv")), index=False)
