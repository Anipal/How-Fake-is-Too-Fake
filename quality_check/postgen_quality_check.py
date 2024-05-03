import torch
from PIL import Image
from utils import *
from tqdm import tqdm
import pdb
from tqdm import tqdm
import csv

torch.set_default_tensor_type('torch.cuda.FloatTensor') 

CSV_READ_FILE = '/raid/mphute6/HalfTruths/Half_Truths_Dataset/full_inpainted_data_sample.csv'
column_map ={
    'image_id' :0, 
    'image_path':1, 
    'dataset':2, 
    'entities':3, 
    'class':4, 
    'mask_id':5, 
    'mask_path':6, 
    'mask_name':7, 
    'ratio':8, 
    'perturbed_image_id':9, 
    'method':10, 
    'model':11, 
    'label':12, 
    'perturbed_mask_name':13, 
    'perturbed_image':14, 
    'image':15, 
    'mask':16, 
    'img_id':17, 
    'classes':18, 
    'sem_magnitude':19, 
    'quality_flag':20, 
    'perturbed_label':21,
    'perturbed_path':22
    }

PATH_TO_DATA_PARENT = "/raid/mphute6/HalfTruths/"
CSV_POSTGEN_QC = "/raid/mphute6/HalfTruths/postgen_quality_check.csv"

def main():

    
    f = open(CSV_READ_FILE, 'r')
    file =  csv.reader(f)
    header = next(file)

    header_qc = header
    header_qc.extend(['cap2_img2', 'direct_sim'])

    out_f = open(CSV_POSTGEN_QC, 'w')
    writer = csv.writer(out_f)
    writer.writerow(header_qc)
    
    for row in tqdm(file):
        if len(row[column_map['image_path']]):
            method = row[column_map['method']]
            model = row[column_map['model']]

            orig_img_path = PATH_TO_DATA_PARENT + row[column_map['image_path']]
            orig_img = Image.open(orig_img_path)
            orig_caption = row[column_map['mask_name']]

            perturbed_img = row[column_map['perturbed_image']]

            ### DEBUG LINE, LEAVE COMMENTED
            # perturbed_img = orig_img

            perturbed_caption = row[column_map['perturbed_mask_name']]

            cap2_img2 = calculate_image_caption_clip_similarity(perturbed_img , perturbed_caption)
            direct_sim = calculate_directional_similarity(orig_img , orig_caption , perturbed_img , perturbed_caption)

            row.extend([cap2_img2, direct_sim])

            writer.writerow(row)

    f.close()
    out_f.close()
    

if __name__ == "__main__":
    main()
