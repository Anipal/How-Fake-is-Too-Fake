#imports
import os
import csv
import random
import pdb

#add path to environment
import sys
sys.path.append("/raid/mphute6/HalfTruths")
sys.path.append("/raid/mphute6/HalfTruths/quality_check")
from quality_check.postgen_quality_check_size import *

#define variables

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

ds_models = {
    "ADE20K":{
        "StableDiffusion_XL"
    },
    "CelebAHQ":{

    },
    "CityScapes":{
        "Kandinsky_2_2",
        "OpenJourney",
        "StableDiffusion_v4",
        "StableDiffusion_v5",
        "StableDiffusion_XL"
    },
    "HumanParsing":{
        "Kandinsky_2_2",
        "OpenJourney",
        "StableDiffusion_v4",
        "StableDiffusion_v5",
        "StableDiffusion_XL"
    },
    "OpenImages":{
        "StableDiffusion_XL"
    },
    "SUN_RGBD":{
        "Kandinsky_2_2",
        "OpenJourney",
        "StableDiffusion_v4",
        # "StableDiffusion_v5",
        "StableDiffusion_XL"
    }
}

#PATHS
PATH_TO_DATA_PARENT_IMG = "/raid/mphute6/HalfTruths/Half_Truths_Dataset/images/"
PATH_TO_DATA_PARENT_CSV = "/raid/mphute6/HalfTruths/inpainting_data/half-truths/half-truths/"
PATH_TO_PERTURB_DATA_PARENT = "/raid/mphute6/HalfTruths/inpainting_data/half-truths/half-truths/"
EDIT_EXTENSION = ".png"
SAMPLED_DATA_PATH = "/raid/mphute6/HalfTruths/random_eval/"
DS_MODEL_PAIR_NUM = 1000

DS_extension_formats ={
        "ADE20K": ".jpg",
        "CelebAHQ": ".jpg",
        "CityScapes": ".png",
        "HumanParsing": ".png",
        "OpenImages": ".jpg",
        "SUN_RGBD": ".jpg",
    }

# Function to get random subset of 2k from each DS-model pair

def get_random_subset(PATH_TO_DATA_PARENT_CSV,
                       ds_models,
                       SAMPLED_DATA_PATH,
                       DS_MODEL_PAIR_NUM):
    pdb.set_trace()
    for ds in ds_models:
        for model in ds_models[ds]:
            print(ds, model)

            orig_subset =os.path.join(SAMPLED_DATA_PATH, "orig",ds, model)
            perturbed_subset =os.path.join(SAMPLED_DATA_PATH, "perturbed", ds,model)
            
            os.makedirs(orig_subset, exist_ok=True)
            os.makedirs(perturbed_subset, exist_ok=True)

            os.system(f"rm -r {orig_subset}/*")
            os.system(f"rm -r {perturbed_subset}/*")

            # Read the csv file
            csv_file = os.path.join(PATH_TO_DATA_PARENT_CSV, ds, f"{ds}_{model}_meta.csv")
            f = open(csv_file, 'r')
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
            f.close()

            # Get random subset of 2k
            random_rows = random.sample(rows, min(DS_MODEL_PAIR_NUM, len(rows)))
            print(len(random_rows))

            # Write the random subset to a new csv file
            random_csv_file = os.path.join(SAMPLED_DATA_PATH, f"{ds}_{model}_sample.csv")

            for i in range(len(random_rows)):
                    row = random_rows[i]
                    orig_img_path = os.path.join(PATH_TO_DATA_PARENT_IMG, ds, row[column_map['img_id']]+DS_extension_formats[ds])
                    perturbed_img_path = os.path.join(PATH_TO_PERTURB_DATA_PARENT , row[column_map['dataset']], model ,row[column_map['mask_id']]+"_"+ds +"_"+model + EDIT_EXTENSION)
                    while not (os.path.exists(orig_img_path) and os.path.exists(perturbed_img_path)):
                        new_row = random.sample(rows, 1)
                        # print(i)
                        if new_row in random_rows:
                             continue
                        else:
                            row = new_row
                        orig_img_path = os.path.join(PATH_TO_DATA_PARENT_IMG, ds, row[column_map['img_id']]+DS_extension_formats[ds])
                        perturbed_img_path = os.path.join(PATH_TO_PERTURB_DATA_PARENT , row[column_map['dataset']], model ,row[column_map['mask_id']]+"_"+ds +"_"+model + EDIT_EXTENSION)
                        random_rows[i] = row 
                        # copy images to subset folder
                    
                    os.system(f"cp {orig_img_path} {orig_subset}")
                    os.system(f"cp {perturbed_img_path} {perturbed_subset}")

            #print number of images in orig and perturbed subset

            print("Number of rows in the csv file", len(random_rows))

            print("Number of original images copied " ,len(os.listdir(orig_subset)))
            print("Number of perturned images copied", len(os.listdir(perturbed_subset)))
                      

            with open(random_csv_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(random_rows)
            
                     

#function to apply post generation_quality_check on the sampled data
def qual_check_random_subset(
                            ds_models,
                            SAMPLED_DATA_PATH,
                            column_map,
                            DS_extension_formats,
                            EDIT_EXTENSION
                        ):
    for ds in ds_models:
        for model in ds_models[ds]:
            print(ds, model)
            csv_file = os.path.join(SAMPLED_DATA_PATH, f"{ds}_{model}_sample.csv")
            os.makedirs(os.path.join(SAMPLED_DATA_PATH,"postgen_qc"), exist_ok=True)
            csv_postgen_qc = os.path.join(SAMPLED_DATA_PATH,"postgen_qc", f"{ds}_{model}_postgen_qc.csv")
            orig_img_path = os.path.join(SAMPLED_DATA_PATH, "orig")
            perturb_img_path = os.path.join(SAMPLED_DATA_PATH, "perturbed")
            DS_EXTENSION = DS_extension_formats[ds]
            postgen_quality_check(
                csv_file,
                csv_postgen_qc,
                ds,
                column_map,
                orig_img_path,
                perturb_img_path,
                DS_EXTENSION,
                EDIT_EXTENSION
            )

def main():
    get_random_subset(PATH_TO_DATA_PARENT_CSV,
                       ds_models,
                       SAMPLED_DATA_PATH,
                       DS_MODEL_PAIR_NUM)
    
    
    qual_check_random_subset(
        ds_models,
        SAMPLED_DATA_PATH,
        column_map,
        DS_extension_formats,
        EDIT_EXTENSION
    )

if __name__ == "__main__":
    main()