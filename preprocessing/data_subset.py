from datasets import Dataset, DatasetDict, Image, Value
import glob 
import os
import pdb 
import sys
from tqdm import tqdm 
import pdb
import pandas as pd

#Taken as argument from command line 
dataset = sys.argv[1]

#Change to where your inpainted images are stored
root_dir = '/srv/share4/apal72/half-truths/data/Half_Truths_Dataset/generated_images/mistral_inpainting/'
#Change to your path to metadata_flat.csv
df = pd.read_csv('/srv/share4/apal72/half-truths/data/Half_Truths_Dataset/metadata_flat.csv')


data_paths = glob.glob(root_dir+dataset+'/*/*' )

models = list(set([f.split('/')[-2] for f in data_paths]))

model_unq = [f.split('/')[-2] for f in data_paths]

min_num = len(model_unq)

for i in tqdm(models, total=len(models)): 
    if(model_unq.count(i) <= min_num):
        min_num = model_unq.count(i)
        model = i

main_list = [f.split('/')[-1].split(dataset)[0] for f in glob.glob(root_dir+dataset+'/' + model + '/*' )]
total_list = []
mask_id = []
model_name = []
# pdb.set_trace()
for path in tqdm(data_paths,total=len(data_paths)):
    # if not(model in path):
    if(path.split('/')[-1].split(dataset)[0] in main_list):
        # pdb.set_trace()
        total_list.append(path.split('/')[-1].split('.')[0])
        model_name.append(path.split('/')[-2])
        mask_id.append(path.split('/')[-1].split(dataset)[0].rsplit('_', 1)[0])


class_name = []
image_id = []
image_path = []
entities = []
mask_path = []
ratio = []
mask_name = []
mask_id_final = []
# pdb.set_trace()
for i in tqdm(mask_id,total=len(mask_id)):
    filtered_rows = df[df['mask_id'] == i]
    mask_id_final.append(filtered_rows['mask_id'].values.tolist()[0])
    class_name.append(filtered_rows['class'].values.tolist()[0])
    image_id.append(filtered_rows['image_id'].values.tolist()[0])
    image_path.append(filtered_rows['image_path'].values.tolist()[0])
    entities.append(filtered_rows['entities'].values.tolist()[0])
    mask_path.append(filtered_rows['mask_path'].values.tolist()[0])
    mask_name.append(filtered_rows['mask_name'].values.tolist()[0])
    ratio.append(filtered_rows['ratio'].values.tolist()[0])

# pdb.set_trace()
columns = ['image_id','image_path','dataset','entities','class','mask_id','mask_path','mask_name','ratio', 'perturbed_mask_name', 'method', 'model', 'label']

# Create an empty DataFrame with specified columns
df_perturb = pd.DataFrame(columns=columns)

df_perturb['image_id'] = image_id
df_perturb['image_path'] = image_path
df_perturb['dataset'] = [dataset]*len(image_id)
df_perturb['entities'] = entities
df_perturb['class'] = class_name
df_perturb['mask_id'] = mask_id_final
df_perturb['mask_path'] = mask_path
df_perturb['mask_name'] = mask_name
df_perturb['ratio'] = ratio
df_perturb['perturbed_mask_name'] = total_list
df_perturb['model'] = model
df_perturb['method'] = ['inpainting']*len(image_id)
df_perturb['label'] = ['fake'] * len(image_id)

df_perturb.to_csv(os.path.join(root_dir,(dataset +'_mistral_subset.csv')), index=False)



    
