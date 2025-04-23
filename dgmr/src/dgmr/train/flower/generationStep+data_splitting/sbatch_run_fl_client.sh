#!/bin/bash

#SBATCH -p uri-gpu         # Partition
#SBATCH -G 1               # Number of GPUs
#SBATCH -t 48:00:00        # Job time limit
#SBATCH -o /work/pi_mzink_umass_edu/SPRITE/skillful_nowcasting/output/sbatch_logs/fl_training_logs/DGMR-fl-training-%j.out    # %j will be replaced with the job ID
#SBATCH --mail-type=ALL    # Send a notification when the job starts, stops, or fails; 'ALL' adds job completion email to the notification
#SBATCH --mem 80G

CLIENT_INT=$1


#read -p "Enter Server Domain: " SERVER_DOMAIN

module load conda/latest
cd PycharmProjects/SPRITE/skillful_nowcasting/train/flower/generationStep+data_splitting
export PYTHONPATH=/home/zhexu_umass_edu/PycharmProjects/SPRITE/skillful_nowcasting:/home/zhexu_umass_edu/PycharmProjects/SPRITE
conda activate dgmr-venv-pycharm

python dgmr_fl_client.py --client_id $CLIENT_INT --server_address "$(cat FL_Server_GPU.txt):8080"