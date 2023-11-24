from sentence_transformers import SentenceTransformer, util
import pdb
import json 
import os
import glob
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Color Scheme for Skyscapes and Uavid")
parser.add_argument('--dataset', type=str, default='ade20k',help='root directory to the labels, ClearNoon') 
parser.add_argument('--root_dir', type=str, default='/srv/share4/apal72/LS/LANCE_release',help='root directory to the labels, ClearNoon')  
args = parser.parse_args()


save_dir = os.path.join(args.root_dir, 'outputs',args.dataset, 'change')
if(not(os.path.exists(save_dir))):
    os.makedirs(save_dir)

# pertrub_type = ['adjective','background','object','subject','weather']

edit_caption_path = os.path.join(args.root_dir,'outputs' ,args.dataset, 'edited_captions')
edit_jsons = glob.glob(edit_caption_path+'/*/*.json')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


for edit in tqdm(edit_jsons, total=len(edit_jsons)):
    sentences=[]
    # cosine_sim = []
    # change = []
    with open(edit, 'r') as json_file:
        json_data = json.load(json_file)

    sentences.append(json_data[0]["original_caption"])
    edited_captions = [item["edited_caption"] for item in json_data]
    sentences.extend(edited_captions)
    embeddings = model.encode(sentences)
    
    for j in range(1,len(embeddings)):
        val = util.pytorch_cos_sim(embeddings[0], embeddings[j])
        json_data[j-1]["cosine_similarity"] = val.numpy()[0][0].tolist()
        json_data[j-1]["semantic_change"] = (1-val).numpy()[0][0].tolist()
        # cosine_sim.append(val)
        # change.append(1-val)
    
    # pdb.set_trace()
    save_path = os.path.join(save_dir,edit.split('/')[-1])
    with open(save_path, 'w') as save_file:
        json.dump(json_data, save_file, indent=2)


