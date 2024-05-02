#!/bin/bash

# Path to the Python script
SCRIPT_PATH="main_orig_edits_hermes.py"

# Output directory
OUTPUT_DIR="/srv/hoffman-lab/share4/apal72/half-truths/data/Half_Truths_Dataset/generated_images/prompt-pompt"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

#The dataset you are perturbing
data="OpenImages"

#The diffusion model you are using
ldm="SDv4"

#Path to the main metadata.json file in Half_Truths_Dataset
JSON="/srv/hoffman-lab/share4/apal72/half-truths/data/Half_Truths_Dataset/metadata.json"

#Output directory where images will be saved
OUTPUT_FILE="$OUTPUT_DIR/hermes/$ldm"
mkdir -p "$OUTPUT_FILE"

#Path to caption json
editcap_path="$OUTPUT_DIR/quality_check/edited_captions/$data.json"

#Root path to folder containing Half_Truths_Dataset
IMG_DIR='/srv/hoffman-lab/share4/apal72/half-truths/data'

accelerate launch --num_processes 6 "$SCRIPT_PATH" --dset_name "ImageFolder"  --img_dir $IMG_DIR --json_path $JSON --ldm_type $ldm --lance_path "$OUTPUT_FILE" --dataset $data --editcap_dict_path $editcap_path --load_caption_edits --verbose

