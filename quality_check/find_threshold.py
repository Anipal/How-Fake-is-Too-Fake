import numpy as np
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import os


PATH_TO_PERTURB_DATA_PARENT = "/raid/mphute6/HalfTruths/inpainting_data/half-truths/half-truths/"
PATH_TO_DATA_PARENT_IMG = "/raid/mphute6/HalfTruths/Half_Truths_Dataset/images/"
EDIT_EXTENSION = ".png"
column_map ={
    'image':0,
    'mask':1,
    'img_id':2,
    'mask_id':3,
    'mask_name':4,
    'ratio':5,
    'entities':6,
    'classes':7,
    'perturbed_label':8,
    'sem_magnitude':9,
    'quality_flag':10,
    'dataset':11,
    'method':12,
    'model':13,
    'perturbed_path':14
}


DS_extension_formats ={
        "ADE20K": ".jpg",
        "CelebAHQ": ".jpg",
        "CityScapes": ".png",
        "HumanParsing": ".png",
        "OpenImages": ".jpg",
        "SUN_RGBD": ".jpg",
    }

#import all csv files in directory
# all_files = glob("/raid/mphute6/HalfTruths/quality_check/postgen_qc/*postgen_qc.csv")
all_files = glob("/raid/mphute6/HalfTruths/quality_check/postgen_qc/*.csv")
list_all_df = [pd.read_csv(f) for f in all_files]
combined_df = pd.concat(list_all_df)

metric_name = ['img1_img2', 'cap2_img2', 'brisque_score_orig', 'brisque_score_perturb', 'direct_sim']

# Plot the metrics
# axes_lim = [[0,1] , [0,1] , [0,100] , [0,100] , [0,1]]
# for metric in metric_name:
#     plt.figure()
#     # Plot the metric
#     plt.hist(combined_df[metric].dropna(), bins=100, color='blue')
#     plt.xlim(axes_lim[metric_name.index(metric)])
#     plt.xlabel('Index')
#     plt.ylabel(metric)
#     plt.title(f'{metric} vs Index')
#     # Save the graph
#     plt.savefig(f'/raid/mphute6/HalfTruths/quality_check/{metric}_graph.png')

# On observing brisque score of perturbed images we see two distinct guassian distributions

# BRISQUE
selected_metric = 'brisque_score_perturb'
# data = combined_df[selected_metric].dropna().values.reshape(-1,1)
threshold = [0,70]

filtered = combined_df[combined_df['brisque_score_perturb'] >=threshold[0]]
print("Original_data ", len(filtered))
filtered = filtered[filtered['brisque_score_perturb'] <=threshold[1]]
print("Filtered data after brisque score ", len(filtered))

# selected_metric = 'cap2_img2'
# data = filtered[selected_metric].dropna().values.reshape(-1,1)

# threshold = [np.percentile(data,20),np.percentile(data,95)]
# filtered = filtered[filtered[selected_metric] >= 0]
# print("Original_data ", len(filtered))
# filtered = filtered[filtered[selected_metric] >=threshold[0]]
# filtered = filtered[filtered[selected_metric] <=threshold[1]]
# print(f"Filtered data after {selected_metric} ", len(filtered))

selected_metric = 'img1_img2'
data = combined_df[selected_metric].dropna().values.reshape(-1,1)

threshold = [np.percentile(data,20),np.percentile(data,95)]
filtered = filtered[filtered[selected_metric] >= 0]
print("Original_data ", len(filtered))
filtered = filtered[filtered[selected_metric] >=threshold[0]]
filtered = filtered[filtered[selected_metric] <=threshold[1]]
print(f"Filtered data after {selected_metric} ", len(filtered))

selected_metric = 'direct_sim'
data = combined_df[selected_metric].dropna().values.reshape(-1,1)

threshold = [np.percentile(data,23),np.percentile(data,100)]
filtered = filtered[filtered[selected_metric] >= 0]
print("Original_data ", len(filtered))
filtered = filtered[filtered[selected_metric] >=threshold[0]]
filtered = filtered[filtered[selected_metric] <=threshold[1]]
print(f"Filtered data after {selected_metric} ", len(filtered))
j=0
# for index, row in filtered.iterrows():
#     j+=1
#     if j<20:
#         continue
#     orig_img_path = os.path.join(PATH_TO_DATA_PARENT_IMG, row['dataset'], row['img_id']+DS_extension_formats[row['dataset']])
#     perturbed_img_path = os.path.join(PATH_TO_PERTURB_DATA_PARENT , row['dataset'], row['model'] ,row['mask_id']+"_"+row['dataset'] +"_"+row['model'] + EDIT_EXTENSION)

#     print("Original image : \n" , orig_img_path)
#     print("Perturbed image : \n" , perturbed_img_path)
#     print("Original Caption : \n" , row['mask_name'])
#     print("Perturbed Caption : \n" , row['perturbed_label'])
#     print(f"{selected_metric} of perturbed image : \n" , row[selected_metric])
#     # print(f"Percentile of {selected_metric} of perturbed image : \n" , )

#     print("----------------------------------------------------")
#     if j>40:
#         break

    





