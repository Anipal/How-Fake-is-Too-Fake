#!/bin/bash
#SBATCH --job-name=dino
#SBATCH --output=dino.out
#SBATCH --error=dino.err
#SBATCH -p overcap
#SBATCH -G a40:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 15 
#SBATCH --exclude="nestor,crushinator,chappie,megabot"
#SBATCH --qos long


python run_1.py --model dinov2