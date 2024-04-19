import pandas as pd 
import os 
import shutil
import pdb
df = pd.read_csv('/srv/share4/apal72/half-truths/data/Half_Truths_Dataset/generated_images/mistral_inpainting/mistral_subset_1.csv')


datasets = ['OpenImages', 'ADE20K', 'CityScapes', 'HumanParsing', 'SUN_RGBD']
models = ['Kandinsky_2_2', 'OpenJourney','StableDiffusion_v4','StableDiffusion_v5','StableDiffusion_XL']
dfs = []
# pdb.set_trace()
for d in datasets:
    if(d == 'OpenImages') or (d == 'ADE20K'):
        m = 'StableDiffusion_XL'
        filtered_rows = df[(df['model'] == m) & (df['dataset'] == d)]
        dfs.append(filtered_rows.sample(frac=1).head(100))
        continue
    for m in models:
        filtered_rows = df[(df['model'] == m) & (df['dataset'] == d)]
        dfs.append(filtered_rows.sample(frac=1).head(100))


combined_df = pd.concat(dfs, axis=0, ignore_index=True)
combined_df.to_csv('/srv/share4/apal72/half-truths/data/Half_Truths_Dataset/generated_images/mistral_inpainting/mistral_subset_100.csv')
# pdb.set_trace()
image_names = combined_df['perturbed_mask_name']
model = combined_df['model']
dataset = combined_df['dataset']
main_dir ='/srv/share4/apal72/half-truths/data/Half_Truths_Dataset/generated_images/mistral_inpainting'
save_dir='/srv/share4/apal72/half-truths/data/Half_Truths_Dataset/generated_images/mistral_inpainting_subset'

for i in range(len(image_names)):
    image_path = os.path.join(main_dir, dataset[i], model[i], (image_names[i]+'.png'))
    save_path = os.path.join(save_dir, dataset[i], model[i])
    if(not(os.path.exists(save_path))):
        os.makedirs(save_path)
    try:
        shutil.copyfile(image_path, (os.path.join(save_path,(image_names[i]+'.png')))) 
    except:
        pass


