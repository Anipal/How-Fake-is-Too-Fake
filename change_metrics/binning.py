import cv2
from sentence_transformers import SentenceTransformer, util
import numpy as np 
import torch
import matplotlib.pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import math
import matplotlib.colors as mcolors
_ = torch.manual_seed(123)
from scipy.ndimage import label
import pandas as pd
import json 
import argparse
import pdb
import os 
from tqdm import tqdm 
from skimage.metrics import structural_similarity
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from caption_generate import *
from calc_lpips import *


def calc_ssim(real,fake):
    # pdb.set_trace()
    score,diff = structural_similarity(real, fake, channel_axis=-1,win_size=5, full=True)
    diff = (diff * 255).astype("uint8")

    return score, diff

def calc_mse(real, fake):
    if(len(real.shape) == 3):
        # mse = np.linalg.norm(real - fake, axis=2)
        mse = np.square(real[:,:,0].astype(np.float32) - fake[:,:,0].astype(np.float32)) + np.square(real[:,:,1].astype(np.float32) - fake[:,:,1].astype(np.float32))+np.square(real[:,:,2].astype(np.float32) - fake[:,:,2].astype(np.float32))
        mse = mse / (255*255*3)
    else:
        mse = np.square(real.astype(np.float32) - fake.astype(np.float32))
        mse = mse/ (255*255)
    mask = (mse*255).astype(np.uint8)
    binary_image = (mask > int(255*0.02)).astype(np.uint8)

    return mse, mask , binary_image

def gen_ratio_dist(all_ratios, path):
    fig, ax = plt.subplots(figsize =(6,4))
    ax.hist(all_ratios, bins=100, color='g')
    plt.title("Distribution of Ratios")
    plt.xlabel("Ratio")
    plt.ylabel("Quantity")
    plt.savefig(path)

def create_bins(ratios, num_bins):
    # Sort the list of ratios
    sorted_ratios = sorted(ratios)
    
    # Calculate the number of ratios per bin
    ratios_per_bin = len(sorted_ratios) // num_bins
    
    # Create bins and calculate min-max for each bin
    bins = {}
    for i in range(num_bins):
        start_index = i * ratios_per_bin
        end_index = (i + 1) * ratios_per_bin
        bin_label = f"Bin_{i+1}"
        bins[bin_label] = {'min_ratio': sorted_ratios[start_index], 'max_ratio': sorted_ratios[end_index - 1]}
    
    return bins

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Color Scheme for Skyscapes and Uavid")
    parser.add_argument('--root_csv', type=str, default='/srv/share4/apal72/half-truths/csvs/consolidated_meta_no_helen.csv',help='main csv') 
    parser.add_argument('--root_dir', type=str, default='/srv/share4/apal72/half-truths/data',help='root directory to the images')  
    parser.add_argument('--save_dir', type=str, default='/srv/share4/apal72/half-truths/data/mse_plots',help='path to save_dir')  
    parser.add_argument('--csv_dir', type=str, default='/srv/share4/apal72/half-truths/csvs',help='path to saving csv')  
    parser.add_argument('--model', type=str, default='lance',help='change model')  

    args = parser.parse_args()  

    ##Initializing the data
    data = pd.read_csv(args.root_csv)
    root_dir = args.root_dir

    if(root_dir.split('/'[-1] == '')):
        data['image_path'] = root_dir + '/'+ data['image_path']
    else:
        data['image_path'] = root_dir + data['image_path']

    data_ = data[data['model'] ==args.model]
    data_id = set(data_['id'])

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #Initialzing the models 
    captioner = CaptionGenerator(device=device, model_type = args.model_type)
    lpips = LPIPS()

    #Iterating over all the datapoints with a given id
    for id in tqdm(data_id,total=len(data_id)):
        #Reading paths from dataframe
        method = data.loc[(data['id'] == id) & (data['label'] == 'fake'), 'method'].values
        fake_img_paths = data.loc[(data['model'] == args.model) & (data['id'] == id) & (data['label'] == 'fake'), 'image_path'].values
        fake_captions_before = data.loc[(data['model'] == args.model) & (data['id'] == id) & (data['label'] == 'fake'), 'perturbed'].values

        # if(data.loc[(data['model'] == args.model) & (data['id'] == i) & (data['label'] == 'fake'), 'dataset'].values[0] =='CelebA-HQ'):
        #     i = "CelebA_{:05d}".format(int(i))
        real_img_path = data.loc[(data['id'] == i) & (data['label'] == 'real'), 'image_path'].values[0]
        real_captions_before = data.loc[(data['model'] == args.model) & (data['id'] == id) & (data['label'] == 'fake'), 'original'].values[0]
        for fake_img_path in fake_img_paths:

            #Calculating semantic similarity in caption space
            real_caption_after, fake_caption_after, sem_sim_caption_after = captioner.generate(real_img_path, fake_img_path)
            sem_sim_caption_before = captioner.cos_sim(real_captions_before, fake_captions_before)

            #Reading images RGB
            img_real_rgb = cv2.imread(real_img_path)
            img_fake_rgb = cv2.imread(fake_img_path)
            if not(img_fake_rgb.shape[:2] == img_real_rgb.shape[:2]):
                img_fake_rgb = cv2.resize(img_fake_rgb, (img_real_rgb.shape[1], img_real_rgb.shape[0]))

            #Images grayscale
            img_real_gray = cv2.cvtColor(img_real_rgb, cv2.COLOR_BGR2GRAY)
            img_fake_gray = cv2.cvtColor(img_fake_rgb, cv2.COLOR_BGR2GRAY)

            #LPIPS score
            lpips_score = lpips.calc_lpips(img_real_rgb, img_fake_rgb)

            #MSE Map
            mse_all_ch, mask_all_ch, binary_image_all_ch = calc_mse(img_real_rgb, img_fake_rgb)
            mse_gray , mask_gray, binary_image_gray = calc_mse(img_real_gray, img_fake_gray)    

            ##SSIM Score
            ssim_score, diff = calc_ssim(img_real_rgb, img_fake_rgb)

            all_data.append([sem_sim_caption_before, sem_sim_caption_after, lpips_score, np.mean(mse_all_ch.flatten()),np.mean(mse_grey.flatten()),ssim_score.item()])        



column_names = ['sen_sim_before', 'sen_sim_after', 'lpips','mse_all','mse_gray','ssim']
df = pd.DataFrame(all_data, columns=column_names)
#Creating different bins 
df_bins = create_even_bins(df)
merged_df = pd.concat([data, df_bins], axis=1)
merged_df.to_csv(os.path.join(args.csv_dir,'all_metrics_bins.csv')




