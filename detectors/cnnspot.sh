#!/bin/bash
#SBATCH --job-name=cnnspot
#SBATCH --output=cnnspot.out
#SBATCH --error=cnnspot.err
#SBATCH -p overcap
#SBATCH -G a40:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 15 
#SBATCH --exclude="nestor,crushinator,chappie,megabot"
#SBATCH --qos long


python run_1.py --model cnnspot

