#!/bin/bash

#SBATCH -p uri-gpu         # Partition
#SBATCH -G 1               # Number of GPUs
#SBATCH --constraint=a100 # Request access to an a100 GPU
#SBATCH --mem 200G
#SBATCH -t 48:00:00        # Job time limit
#SBATCH -o /work/pi_mzink_umass_edu/SPRITE/outputs/ldcast/diffusion/VTPM/ldcast_diffusion-training-%j.out    # %j will be replaced with the job ID
#SBATCH --mail-type=ALL    # Send a notification when the job starts, stops, or fails; 'ALL' adds job completion email to the notification

#read -p "Enter Server Domain: " SERVER_DOMAIN

source ../.venv/bin/activate

cd /home/zhexu_umass_edu/PycharmProjects/SPRITE/ldcast/src/ldcast

python -m dotenv run -- python scripts/train_genforecast.py "config/train_config_diffusion.yaml" --job_id=${SLURM_JOB_ID}

deactivate