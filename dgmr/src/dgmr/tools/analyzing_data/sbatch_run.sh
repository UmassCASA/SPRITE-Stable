#!/bin/bash

#SBATCH -p uri-gpu         # Partition
#SBATCH -G 1               # Number of GPUs
#SBATCH --mem 200G
#SBATCH -c 32
#SBATCH -t 48:00:00        # Job time limit 
#SBATCH -o /home/zhexu_umass_edu/PycharmProjects/SPRITE/skillful_nowcasting/DGMRTools/analyzing_data/data_distribution_log-%j.out    # %j will be replaced with the job ID
#SBATCH --mail-type=ALL    # Send a notification when the job starts, stops, or fails; 'ALL' adds job completion email to the notification

# Load the necessary module
module load conda/latest

# Activate the conda environment
# conda activate dgmr-venv
conda activate dgmr-venv-pycharm

# Change to the appropriate directory
cd /home/zhexu_umass_edu/PycharmProjects/SPRITE/skillful_nowcasting/DGMRTools/analyzing_data

export PYTHONPATH=/home/zhexu_umass_edu/PycharmProjects/SPRITE/skillful_nowcasting:/home/zhexu_umass_edu/PycharmProjects/SPRITE

# Run the Python script
python plotting_data_feature.py
# python3 train/run_casa.py

# Deactivate the conda environment
conda deactivate

