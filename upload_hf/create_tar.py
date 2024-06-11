import os
import tarfile
import shutil
from glob import glob
from tqdm import tqdm

# Path to the directory containing the data
data_paths = "/raid/mphute6/HalfTruths/inpainting_data/half-truths/half-truths/"

#Name of dataset
all_datasets = {
    "CityScapes":[
        "Kandinsky_2_2", 
          "OpenJourney", 
          "StableDiffusion_v4",
        # "StableDiffusion_v5",
        # "StableDiffusion_XL",
    ],
    # "HumanParsing":
    # [
    #     "Kandinsky_2_2", 
    #       "OpenJourney", 
    #       "StableDiffusion_v4",
    #        "StableDiffusion_v5", 
    #        "StableDiffusion_XL"
    # ],
    # "OpenImages":
    # [
    #     "StableDiffusion_XL",
    # ],
    # "SUN_RGBD":
    # [
    #     "Kandinsky_2_2", 
    #       "OpenJourney", 
    #       "StableDiffusion_v4",
    #        "StableDiffusion_v5", 
    #        "StableDiffusion_XL"
    # ]
}

# # Diffusion models
# models = [
#           "Kandinsky_2_2", 
#           "OpenJourney", 
#           "StableDiffusion_v4",
#            "StableDiffusion_v5", 
#            "StableDiffusion_XL"
#            ]



# Create a tar file for each model
for dataset in all_datasets:
    for model in all_datasets[dataset]:
        model_path = os.path.join(data_paths,dataset, model)
        model_files = glob(model_path + "/*")
        num_files = len(model_files)
        for i in range(3):
            tar_name = f"{dataset}_{model}_{i}.tar.bz2"
            model_subset = model_files[i*int(num_files/3):(i+1)*int(num_files/3)]
            with tarfile.open(tar_name, "w:bz2") as tar:
                for i in tqdm(range(len(model_subset))):
                    tar.add(model_subset[i], arcname=os.path.basename(model_subset[i]))
            shutil.move(tar_name, data_paths)
            print(f"Created tar file for {model}")