#!/bin/bash

#SBATCH -c 1  # Number of Cores per Task
#SBATCH -p cpu-long  # Partition
#SBATCH -t 48:00:00  # Job time limit
#SBATCH -o /work/pi_mzink_umass_edu/SPRITE/skillful_nowcasting/output/sbatch_logs/preprocessing_logs/max_nimrod-%j.out  # %j = job ID
#SBATCH --mail-type=ALL    # Send a notification when the job starts, stops, or fails; 'ALL' adds job completion email to the notification
#SBATCH --mem-per-cpu=8G



# Load the necessary module
module load miniconda/22.11.1-1

# Activate the conda environment
# conda activate dgmr-venv
conda activate dgmr-casa-venv

# Change to the appropriate directory
cd /work/pi_mzink_umass_edu/SPRITE/skillful_nowcasting

# Run the Python script
python3 WORK/extract_info/max_values_nimrod.py

# Deactivate the conda environment
conda deactivate