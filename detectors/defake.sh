#!/bin/bash
#SBATCH --job-name=all_ade_sdv5
#SBATCH --output=all_ade_sdv5.out
#SBATCH --error=all_ade_sdv5.err
#SBATCH -p hoffman-lab
#SBATCH -G a40:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 15 
#SBATCH --exclude="nestor,crushinator,chappie,megabot"
#SBATCH --qos long

# Path to the Python script
SCRIPT_PATH="main_orig_edits.py"

# Output directory
OUTPUT_DIR="/srv/hoffman-lab/share4/apal72/half-truths/data/Half_Truths_Dataset/generated_images/prompt-pompt"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Generate values for the variables
# VALUES=$(awk 'BEGIN{ for(i=0.6;i<=0.9;i+=0.1) print i }')
# VALUES_1=$(awk 'BEGIN{ for(i=0.0;i<=0.5;i+=0.1) print i }')
# VALUES_2=$(awk 'BEGIN{ for(i=0.8;i<=0.95;i+=0.1) print i }')
# val4=0.7
data="ADE20K"
ldm="SDv5"
# JSON="/srv/share4/apal72/half-truths/data/Half_Truths_Dataset/metadata/_llama_sub_${data}_3.json"
JSON="/srv/share4/apal72/half-truths/data/Half_Truths_Dataset/metadata/_llama_${ldm}_sub_${data}_4.json"
OUTPUT_FILE="$OUTPUT_DIR/$ldm"
mkdir -p "$OUTPUT_FILE"
# cap_root="$OUTPUT_DIR/captions/$data/"
# mkdir -p "$cap_root"
editcap_path="$OUTPUT_DIR/captions/$data/editcap_final_dict_2.json"
gencap_path="$OUTPUT_DIR/captions/$data/gencap_dict.json"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python "$SCRIPT_PATH" --dset_name "ImageFolder"  --img_dir '/srv/hoffman-lab/share4/apal72/half-truths/data' --json_path '/srv/share4/apal72/half-truths/data/Half_Truths_Dataset/metadata/_llama_sdv4_sub_ADE20K.json' --ldm_type $ldm --lance_path "$OUTPUT_FILE" --dataset $data --editcap_dict_path $editcap_path --gencap_dict_path $gencap_path --load_captions --load_caption_edits --verbose

accelerate launch --num_processes 6 "$SCRIPT_PATH" --dset_name "ImageFolder"  --img_dir '/srv/hoffman-lab/share4/apal72/half-truths/data' --json_path $JSON --ldm_type $ldm --lance_path "$OUTPUT_FILE" --dataset $data --editcap_dict_path $editcap_path --gencap_dict_path $gencap_path --load_captions --load_caption_edits --verbose

