#!/bin/bash
#SBATCH --job-name=dire
#SBATCH --output=dire.out
#SBATCH --error=dire.err
#SBATCH -p overcap
#SBATCH -G a40:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 15 
#SBATCH --exclude="nestor,crushinator,chappie,megabot"
#SBATCH --qos long


python run_1.py --model dire