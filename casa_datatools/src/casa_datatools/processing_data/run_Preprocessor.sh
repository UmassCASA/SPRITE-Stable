#!/bin/bash

#SBATCH -c 16  # Number of Cores per Task
#SBATCH -p cpu  # Partition
#SBATCH -t 48:00:00  # Job time limit
#SBATCH -o /work/pi_mzink_umass_edu/SPRITE/outputs/DATA/processing_data_logs/Preprocess-Seq-%j.txt  # %j = job ID
#SBATCH --mail-type=ALL    # Send a notification when the job starts, stops, or fails; 'ALL' adds job completion email to the notification


source /work/pi_mzink_umass_edu/venvs/uv-SPRITE/.venv-a/bin/activate
export PYTHONPATH=/work/pi_mzink_umass_edu/SPRITE/dgmr/src:/work/pi_mzink_umass_edu/SPRITE:/work/pi_mzink_umass_edu/SPRITE/nowcastnet/src:/work/pi_mzink_umass_edu/SPRITE/smaat_unet/src:/work/pi_mzink_umass_edu/SPRITE/casa_datatools/src

# Change to the appropriate directory
cd /work/pi_mzink_umass_edu/SPRITE/casa_datatools

# Run the Python script
python src/casa_datatools/processing_data/run_Preprocessor.py

# Deactivate the conda environment
deactivate