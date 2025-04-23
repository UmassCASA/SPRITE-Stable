#!/bin/bash

#SBATCH -c 32  # Number of Cores per Task
#SBATCH -p uri-gpu         # Partition
#SBATCH -G 1               # Number of GPUs
#SBATCH --mem 200G
#SBATCH -t 48:00:00        # Job time limit
#SBATCH -o /home/zhexu_umass_edu/PycharmProjects/SPRITE/model_compare/pysteps-experiment-%j.out    # %j will be replaced with the job ID
#SBATCH --mail-type=ALL    # Send a notification when the job starts, stops, or fails; 'ALL' adds job completion email to the notification

# Load the necessary module
module load conda/latest

# Activate the conda environment
# conda activate dgmr-venv
conda activate dgmr-venv-pycharm

# Change to the appropriate directory
cd /home/zhexu_umass_edu/PycharmProjects/SPRITE/model_compare

export PYTHONPATH=/home/zhexu_umass_edu/PycharmProjects/SPRITE/skillful_nowcasting:/home/zhexu_umass_edu/PycharmProjects/SPRITE:/home/zhexu_umass_edu/PycharmProjects/SPRITE/NowcastNet/code
#conda activate dgmr-venv-pycharm

# Run the Python script
python3 run_experiment.py --incremental
#python run_experiment.py --incremental  --dates '{"idx: 148": 148}'
#python3 run_experiment.py
# python3 train/run_casa.py

# Deactivate the conda environment
conda deactivate
