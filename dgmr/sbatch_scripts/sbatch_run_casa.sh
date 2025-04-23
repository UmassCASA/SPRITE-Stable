#!/bin/bash

#SBATCH -p uri-gpu         # Partition
#SBATCH -G 1               # Number of GPUs
#SBATCH --mem 100G
#SBATCH --constraint=a100 # Request access to an a100 GPU
#SBATCH -t 48:00:00        # Job time limit 
#SBATCH -o /work/pi_mzink_umass_edu/SPRITE/skillful_nowcasting/output/sbatch_logs/model_training_logs/DGMR-CASA-SingleGPU_16precision_6gen_steps-%j.out    # %j will be replaced with the job ID
#SBATCH --mail-type=ALL    # Send a notification when the job starts, stops, or fails; 'ALL' adds job completion email to the notification

# Load the necessary module
module load miniconda/22.11.1-1

# Activate the conda environment
# conda activate dgmr-venv
conda activate dgmr-casa-venv

# Change to the appropriate directory
cd /work/pi_mzink_umass_edu/SPRITE/skillful_nowcasting

# Run the Python script
# python3 train/run.py
python3 train/run_casa.py

# Deactivate the conda environment
conda deactivate

