#!/bin/bash

#SBATCH -p uri-gpu         # Partition
#SBATCH -G 1               # Number of GPUs
#SBATCH -t 48:00:00        # Job time limit
#SBATCH -o /work/pi_mzink_umass_edu/SPRITE/skillful_nowcasting/output/sbatch_logs/fl_training_logs/DGMR-fl-training-%j.out    # %j will be replaced with the job ID
#SBATCH --mail-type=ALL    # Send a notification when the job starts, stops, or fails; 'ALL' adds job completion email to the notification
#SBATCH --mem 200G

#read -p "Enter Server Domain: " SERVER_DOMAIN
CLIENT_COUNT=$1

module load conda/latest
cd PycharmProjects/SPRITE/skillful_nowcasting/train/flower/generationStep+data_splitting
export PYTHONPATH=/home/zhexu_umass_edu/PycharmProjects/SPRITE/skillful_nowcasting:/home/zhexu_umass_edu/PycharmProjects/SPRITE
conda activate dgmr-venv-pycharm

hostname -f > FL_Server_GPU.txt

python dgmr_fl_server.py --server_address "0.0.0.0:8080" --num_rounds 10000 --min_fit_clients $CLIENT_COUNT --min_available_clients $CLIENT_COUNT --client_number $CLIENT_COUNT
