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
from clip_similarity import *
from dreamsim import dreamsim
from transformers import BlipProcessor, BlipForConditionalGeneration 
from PIL import Image, ImageOps 





class Change:
    def __init__(
        self,
        perturb_type,
        mse_rgb_threshold: int = 0.5,
        mse_gray_threshold: int = 0.5,
        blip_model_name: str = "Salesforce/blip-image-captioning-large",
        sentence_transormer_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        clip_model_name: str = "ViT-L/14",
        device: str = "cuda",
    ):

        self.mse_rgb_threshold = mse_rgb_threshold
        self.mse_gray_threshold = mse_gray_threshold
        self.perturb_type = perturb_type
        self.device = device

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
        self.clip = ClipSimilarity(clip_model_name, device)
        self.dreamsim_model, self.dreamsim_preprocess = dreamsim(pretrained=True)
        self.sen_sim = SentenceTransformer(sentence_transormer_model_name)
        self.caption_model = BlipForConditionalGeneration.from_pretrained(blip_model_name)
        self.caption_processor = BlipProcessor.from_pretrained(blip_model_name)
    


    def normalize_lpips(self, arr):
        # pdb.set_trace()
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        arr_range = arr_max - arr_min
        scaled = np.array((arr-arr_min) / float(arr_range), dtype='f')
        arr_new = -1 + (scaled * 2)
        return arr_new


    def get_blip_caption(self, fake):
        # pdb.set_trace()
        text = "a photo of"
        inputs = self.caption_processor(fake,text, return_tensors="pt").to(self.device)
        out = self.caption_model.generate(**inputs)
        caption_og = self.caption_processor.decode(out[0], skip_special_tokens=True)
        return caption_og

    def calc_embeddings(self,fake_img_path='',original_caption='', edited_caption='' ):
        # pdb.set_trace()
        if(self.perturb_type == 'p2p'):
            caption_json_path = fake_img_path.replace(os.path.join(fake_img_path.split('/')[-3],fake_img_path.split('/')[-2],fake_img_path.split('/')[-1]), os.path.join('edited_captions',fake_img_path.split('/')[-3],(fake_img_path.split('/')[-2]+'.json')))
            with open(caption_json_path, 'r') as file:
                caption_json = json.load(file)

            for caption in caption_json:
                if(caption['edit'] ==fake_img_path.split('/')[-1].split('.')[0]):
                    original_caption = caption['original_caption']
                    edited_caption = caption['edited_caption']
                    break
        sentences = [original_caption, edited_caption]  
        return sentences, original_caption


    def calc_sen_sim(self,sentences):
        # pdb.set_trace()type(real)
        embeddings = self.sen_sim.encode(sentences)
        val = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        sem_sim_caption = val.numpy()[0][0]

        return sem_sim_caption

    def calc_ssim(self,real,fake):
        # pdb.set_trace()
        # pdb.set_trace()
        real = np.array(real)
        fake = np.array(fake)
        score,diff = structural_similarity(real, fake, channel_axis=-1,win_size=5, full=True)
        # diff = (diff * 255).astype("uint8")

        return score

    def calc_mse(self,real,fake):
        # pdb.set_trace()
        real = np.array(real)
        fake = np.array(fake)
        if(len(real.shape) == 3):
            mse = np.linalg.norm(real - fake, axis=2)
            mse = np.square(real[:,:,0].astype(np.float32) - fake[:,:,0].astype(np.float32)) + np.square(real[:,:,1].astype(np.float32) - fake[:,:,1].astype(np.float32))+np.square(real[:,:,2].astype(np.float32) - fake[:,:,2].astype(np.float32))
            mse = mse / (255*255*3)
        else:
            mse = np.square(real.astype(np.float32) - fake.astype(np.float32))
            mse = mse/ (255*255)
        # mask = (mse*255).astype(np.uint8)
        # binary_image = (mask > int(255*0.02)).astype(np.uint8)

        return np.mean(mse.flatten())
    
    def calc_dreamsim(self, real, fake):
        # pdb.set_trace()
        real = self.dreamsim_preprocess(real).to(self.device)
        fake = self.dreamsim_preprocess(fake).to(self.device)
        distance = self.dreamsim_model(real, fake)

        return distance.item()

    
    def calc_metrics(self,real_img_path,fake_img_path, og_caption ='',edit_caption =''):
        # pdb.set_trace()
        img_real_rgb = Image.open(real_img_path).convert('RGB')
        img_fake_rgb = Image.open(fake_img_path).convert('RGB')
        img_fake_rgb =img_fake_rgb.resize(img_real_rgb.size)

        img_real_gray=ImageOps.grayscale(img_real_rgb)
        img_fake_gray=ImageOps.grayscale(img_fake_rgb)

        #Semantic Image Space
        dreamsim = self.calc_dreamsim(img_real_rgb, img_fake_rgb)
        lpips_score = (self.lpips(torch.from_numpy(self.normalize_lpips(img_real_rgb)).permute(2, 0, 1).unsqueeze(0), torch.from_numpy(self.normalize_lpips(img_fake_rgb)).permute(2, 0, 1).unsqueeze(0))).item()

        #Semantic Caption space
        if(self.perturb_type =='p2p'):
            sentences, og_caption = self.calc_embeddings(fake_img_path=fake_img_path)
        else:
            sentences, og_caption = self.calc_embeddings(original_caption=og_caption, edited_caption=edit_caption)
        sen_sim = self.calc_sen_sim(sentences)
        clip_sim = (self.clip.text_similarity(sentences[0], sentences[1])).item()
        # post_caption = self.get_blip_caption(img_fake_rgb)
        # sentences_post = [og_caption, post_caption]
        # sen_sim_post = self.calc_sen_sim(sentences_post)
        
        #Image size space
        mse_rgb = self.calc_mse(img_real_rgb, img_fake_rgb)
        mse_gray = self.calc_mse(img_real_gray, img_fake_gray)
        ssim_rgb = self.calc_ssim(img_real_rgb,img_fake_rgb)
        ssim_gray = self.calc_ssim(img_real_gray,img_fake_gray)

        return [dreamsim, lpips_score, sen_sim, clip_sim, mse_rgb, mse_gray, ssim_rgb, ssim_gray]










