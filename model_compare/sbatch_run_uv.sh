# ============ General Settings ============
gpus=1                # Number of GPUs per node
partition="uri-gpu"       # SLURM partition (e.g., gpu, uri-gpu, gpu-preempt)
constraint="a100-80g"   # GPU type

project="Model_Compare" # Project name for outputs and logging
base_output_dir="/work/pi_mzink_umass_edu/SPRITE/outputs"

sbatch <<EOT 
#!/bin/bash

#SBATCH --job-name=${project}
#SBATCH -p ${partition}
#SBATCH -c 32  
#SBATCH --constraint=${constraint}
#SBATCH --gres=gpu:${gpus}
#SBATCH --ntasks-per-node=${gpus}
#SBATCH -t 100:00:00
#SBATCH -o ${base_output_dir}/${project}/logs/log-J%j.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=200G

# Job-specific output directory
output_dir="${base_output_dir}/${project}/logs"
mkdir -p "\$output_dir"

# Activate the virtual senvironment
source /work/pi_mzink_umass_edu/SPRITE/venvs/sprite-model-compare-venv/bin/activate

export PYTHONPATH=/work/pi_mzink_umass_edu/SPRITE/skillful_nowcasting:/work/pi_mzink_umass_edu/SPRITE:/work/pi_mzink_umass_edu/SPRITE/NowcastNet/code:/work/pi_mzink_umass_edu/SPRITE/SmaAt_UNet:/work/pi_mzink_umass_edu/SPRITE/CASA_DataTools

cd /work/pi_mzink_umass_edu/SPRITE/model_compare


# Run the Python script
# python3 run_experiment.py --incremental
# python run_experiment.py --incremental  --dates '{"idx: 148": 148}'

python3 model_selecting.py

# Deactivate the conda environment
deactivate

EOT