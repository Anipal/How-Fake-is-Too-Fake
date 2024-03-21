#!/bin/bash
SCRIPT_PATH="main.py"

# Output directory
OUTPUT_DIR="outputs"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="$OUTPUT_DIR/test_3"
mkdir -p "$OUTPUT_FILE"

python "$SCRIPT_PATH" --dset_name "ImageFolder"  --img_dir '/srv/hoffman-lab/share4/apal72/half-truths/data' --json_path '/srv/hoffman-lab/share4/apal72/half-truths/data/Half_Truths_Dataset/metadata.json' --ldm_type 'stable_diffusion_v1_4' --lance_path "$OUTPUT_FILE"

