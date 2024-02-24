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

def normalize(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    print('Min: %.3f, Max: %.3f' % (arr_min, arr_max))
    arr_range = arr_max - arr_min
    scaled = np.array((arr-arr_min) / float(arr_range), dtype='f')
    arr_new = -1 + (scaled * 2)
    print('Min: %.3f, Max: %.3f' % (arr_new.min(), arr_new.max()))
    return arr_new

def calc_sem_sim(fake_img_path, model):
    caption_json_path = fake_img_path.replace(os.path.join(fake_img_path.split('/')[-3],fake_img_path.split('/')[-2],fake_img_path.split('/')[-1]), os.path.join('edited_captions',fake_img_path.split('/')[-3],(fake_img_path.split('/')[-2]+'.json')))
    with open(caption_json_path, 'r') as file:
        caption_json = json.load(file)

    for caption in caption_json:
        if(caption['edit'] ==fake_img_path.split('/')[-1].split('.')[0]):
            original_caption = caption['original_caption']
            edited_caption = caption['edited_caption']
            break
    sentences = [original_caption, edited_caption]
    embeddings = model.encode(sentences)
    val = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    sem_sim_caption = val.numpy()[0][0]

    return sem_sim_caption, original_caption, edited_caption

def calc_ssim(real,fake):
    # pdb.set_trace()
    score,diff = structural_similarity(real, fake, channel_axis=-1,win_size=5, full=True)
    diff = (diff * 255).astype("uint8")

    return score, diff




def calc_mse(real, fake):
    if(len(real.shape) == 3):
        mse = np.linalg.norm(real - fake, axis=2)
        mse = np.square(real[:,:,0].astype(np.float32) - fake[:,:,0].astype(np.float32)) + np.square(real[:,:,1].astype(np.float32) - fake[:,:,1].astype(np.float32))+np.square(real[:,:,2].astype(np.float32) - fake[:,:,2].astype(np.float32))
        mse = mse / (255*255*3)
    else:
        mse = np.square(real.astype(np.float32) - fake.astype(np.float32))
        mse = mse/ (255*255)
    mask = (mse*255).astype(np.uint8)
    binary_image = (mask > int(255*0.02)).astype(np.uint8)

    return mse, mask , binary_image

def plot_cc(labeled_image):
    num_unique_values = np.max(labeled_image) + 1  # Number of unique values including 0
    colors = plt.cm.get_cmap('tab20', num_unique_values - 1)  # Choose a colormap and get unique colors, excluding 0
    black_color = np.array([[0, 0, 0, 1]])  # Black color for label 0
    colors_array = colors(np.arange(num_unique_values - 1))  # Convert colormap object to numpy array
    colors = np.vstack((black_color, colors_array))  # Add black color for label 0
    cmap = mcolors.ListedColormap(colors)
    return cmap


def calc_cc(binary_image):
    if(len(np.unique(binary_image)) < 2):
        return 0, 0, 0, "gray", 0
    num_labels, labeled_image = cv2.connectedComponents(binary_image)
    flattened_image = labeled_image.flatten()
    label_counts = np.bincount(flattened_image)
    largest_component_size = np.max(label_counts[1:])
    centroids = []
    for label_idx in range(1, num_labels):  # Exclude background label (0)
        labeled_indices = np.where(labeled_image == label_idx)
        centroid_x = np.mean(labeled_indices[1])
        centroid_y = np.mean(labeled_indices[0])
        centroids.append((centroid_x, centroid_y))
    mean_centroid = np.mean(centroids, axis=0)
    std_centroid = np.std(centroids, axis=0)
    cmap = plot_cc(labeled_image)
    return largest_component_size, mean_centroid, std_centroid, cmap, labeled_image, num_labels

def save_img(img_real_rgb, img_fake_rgb,img_fake_gray, mse_all_ch, mse_gray,labeled_image_all_ch, labeled_image_gray, cmap_all_ch, cmap_gray, diff, score, path):
    if isinstance(labeled_image_all_ch, int):
        labeled_image_all_ch = np.zeros((mse_all_ch.shape[0],mse_all_ch.shape[1]))
    if isinstance(labeled_image_gray, int):
        labeled_image_gray = np.zeros((mse_gray.shape[0],mse_gray.shape[1]))

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes[0, 0].imshow(cv2.cvtColor(img_real_rgb, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('original image ')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(cv2.cvtColor(img_fake_rgb, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('changed image')
    axes[0, 1].axis('off')
    axes[0, 2].imshow(diff,cmap='hot')
    txt = 'ssim diff, SSIM ='+str(score.item())
    axes[0, 2].set_title(txt)
    axes[0, 2].axis('off')
    # axes[0, 2].imshow(img_fake_gray,cmap="gray")
    # axes[0, 2].set_title('changed image grayscale')
    # axes[0, 2].axis('off')

    im = axes[1, 0].imshow(mse_all_ch, cmap='hot',vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=axes[1, 0])
    cbar.set_label('MSE  ')
    axes[1, 0].legend(['MSE'], loc='upper right')
    axes[1, 0].imshow(mse_all_ch,cmap='hot')
    axes[1, 0].set_title('mse visualization(all channels)')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(labeled_image_all_ch,cmap=cmap_all_ch)
    axes[1, 1].set_title('connected components(all channels) ')
    axes[1, 1].axis('off')
    axes[1, 2].hist(mse_all_ch.flatten())
    axes[1, 2].set_xlabel('mse values(all_channels)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Histogram of all values in mse')

    im = axes[2, 0].imshow(mse_gray, cmap='hot',vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=axes[2, 0])
    cbar.set_label('MSE  ')
    axes[2, 0].legend(['MSE'], loc='upper right')
    axes[2, 0].imshow(mse_gray,cmap='hot')
    axes[2, 0].set_title('mse visualization(grayscale)')
    axes[2, 0].axis('off')
    axes[2, 1].imshow(labeled_image_gray,cmap=cmap_gray)
    axes[2, 1].set_title('connected components(gray) ')
    axes[2, 1].axis('off')
    axes[2, 2].hist(mse_gray.flatten())
    axes[2, 2].set_xlabel('mse values(grayscale)')
    axes[2, 2].set_ylabel('Frequency')
    axes[2, 2].set_title('Histogram of all values in mse')

    plt.tight_layout()
    plt.savefig(path)




parser = argparse.ArgumentParser(description="Color Scheme for Skyscapes and Uavid")
parser.add_argument('--root_csv', type=str, default='/srv/share4/apal72/half-truths/csvs/consolidated_meta_no_helen.csv',help='main csv') 
parser.add_argument('--root_dir', type=str, default='/srv/share4/apal72/half-truths/data',help='root directory to the images')  
parser.add_argument('--save_dir', type=str, default='/srv/share4/apal72/half-truths/data/mse_plots',help='path to save_dir')  
parser.add_argument('--csv_dir', type=str, default='/srv/share4/apal72/half-truths/csvs',help='path to saving csv')  
parser.add_argument('--model', type=str, default='lance',help='change model')  

args = parser.parse_args()      

data = pd.read_csv(args.root_csv)
data_ = data[data['model'] ==args.model]
data_id = set(data_['id'])
root_dir = args.root_dir
model_sem_sim = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

all_data =[]


for i in tqdm(data_id,total=len(data_id)):
    #Reading paths from dataframe
    pertrub = data.loc[(data['id'] == i) & (data['label'] == 'fake'), 'pertrubation_type'].values[0]
    fake_img_path = os.path.join(root_dir,data.loc[(data['model'] == args.model) & (data['id'] == i) & (data['label'] == 'fake'), 'image_path'].values[0])
    if(data.loc[(data['model'] == args.model) & (data['id'] == i) & (data['label'] == 'fake'), 'dataset'].values[0] =='CelebA-HQ'):
        i = "CelebA_{:05d}".format(int(i))
    real_img_path = os.path.join(root_dir,data.loc[(data['id'] == i) & (data['label'] == 'real'), 'image_path'].values[0])
    
    #Calculating semantic similarity in caption space
    sem_sim_caption, original_caption, edited_caption = calc_sem_sim(fake_img_path, model_sem_sim)
    

    #Reading images RGB
    img_real_rgb = cv2.imread(real_img_path)
    img_fake_rgb = cv2.imread(fake_img_path)
    img_fake_rgb = cv2.resize(img_fake_rgb, (img_real_rgb.shape[1], img_real_rgb.shape[0]))

    #Images grayscale
    img_real_gray = cv2.cvtColor(img_real_rgb, cv2.COLOR_BGR2GRAY)
    img_fake_gray = cv2.cvtColor(img_fake_rgb, cv2.COLOR_BGR2GRAY)

    #LPIPS score
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
    score = lpips(torch.from_numpy(normalize(img_real_rgb)).permute(2, 0, 1).unsqueeze(0), torch.from_numpy(normalize(img_fake_rgb)).permute(2, 0, 1).unsqueeze(0))
    
    #MSE Map
    mse_all_ch, mask_all_ch, binary_image_all_ch = calc_mse(img_real_rgb, img_fake_rgb)
    mse_gray , mask_gray, binary_image_gray = calc_mse(img_real_gray, img_fake_gray)

    #Connected components 
    largest_component_size_all, mean_centroid_all_ch, std_centroid_all_ch, cmap_all_ch, labeled_image_all_ch, num_labels_all = calc_cc(binary_image_all_ch)
    largest_component_size_gray, mean_centroid_gray, std_centroid_gray, cmap_gray, labeled_image_gray, num_labels_gray = calc_cc(binary_image_gray)

    #Saving image 
    save_path = os.path.join(args.save_dir,fake_img_path.split('/')[-2], fake_img_path.split('/')[-1])
    if(not(os.path.exists(os.path.join(args.save_dir,fake_img_path.split('/')[-2])))):
        os.makedirs(os.path.join(args.save_dir,fake_img_path.split('/')[-2]))
    # pdb.set_trace()
    ssim_score, diff = calc_ssim(img_real_rgb, img_fake_rgb)
    save_img(img_real_rgb, img_fake_rgb, img_fake_gray,mse_all_ch, mse_gray,labeled_image_all_ch, labeled_image_gray, cmap_all_ch, cmap_gray, diff, ssim_score, save_path)
    all_data.append([fake_img_path, real_img_path, original_caption,edited_caption,pertrub, np.mean(mse_all_ch.flatten()),np.mean(mse_gray.flatten()), score.item(), sem_sim_caption ,mean_centroid_all_ch,std_centroid_all_ch,mean_centroid_gray,std_centroid_gray,largest_component_size_all, largest_component_size_gray,num_labels_all, num_labels_gray,ssim_score.item()])

column_names = ['fake_img_path', 'real_img_path', 'original_caption', 'edited_caption', 'perturbation_type', 'mse_all', 'mse_gray', 'lpips','sen_sim', 'cc_mean_all','cc_std_all','cc_mean_gray','cc_std_gray','cc_largest_all','cc_largest_gray','num_labels_all','num_labels_gray','ssim']
df = pd.DataFrame(all_data, columns=column_names)
df.to_csv(os.path.join(args.csv_dir,'all_changes_lance.csv'))


    



