#!/bin/bash

# ============ General Settings ============
num_nodes=1           # Number of nodes to use
gpus=1                # Number of GPUs per node
partition="uri-gpu,gpu"       # SLURM partition (e.g., gpu, uri-gpu, gpu-preempt)
project="TEST" # Project name for outputs and logging
# constraint="a100-80g,ib"   # GPU type
constraint="l40s"   # GPU type
memory=200G
time=1:00:00

# ============ Training Settings ============
precision="32"             # Precision mode (e.g., "16-mixed", "32")
gen_steps=6                # Number of generation steps
batch_size=16              # Batch size for training
strategy="ddp_find_unused_parameters_true" # DDP strategy
checkpoint_path=""
epochs=1                 # Number of training epochs


# ============ Model Configuration ============
model="NowcastNet–CASA"
base_output_dir="/work/pi_mzink_umass_edu/SPRITE/outputs"
setup="N${num_nodes}-G${gpus}"
configuration="P${precision}_GS${gen_steps}_BS${batch_size}"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${project}
#SBATCH -p ${partition}
#SBATCH --constraint=${constraint}
#SBATCH -N ${num_nodes}
#SBATCH --gres=gpu:${gpus}
#SBATCH --ntasks-per-node=${gpus}
#SBATCH -t ${time}
#SBATCH --mem=${memory}
#SBATCH -o ${base_output_dir}/${project}/J%j-${model}-${setup}-${configuration}/log-J%j.txt
#SBATCH --mail-type=END,FAIL

# Job-specific output directory
output_dir="${base_output_dir}/${project}/J\${SLURM_JOB_ID}-${model}-${setup}-${configuration}"
mkdir -p "\$output_dir"

# Summary
echo "Job Configuration Summary:"
echo "--------------------------"
echo "Project: ${project}"
echo "Model: ${model}"
echo "Configuration: ${configuration}"
echo "Setup: ${setup}"
echo "Output Directory: \$output_dir"
echo ""


# Log GPU details
gpu_info=\$(nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader)
echo "Allocated GPU details:"
echo "\$gpu_info"
echo ""

# Activate the virtual environment
source /work/pi_mzink_umass_edu/venvs/uv-SPRITE/.venv-a/bin/activate

# Add the project directory to the PYTHONPATH
export PYTHONPATH=/work/pi_mzink_umass_edu/SPRITE/dgmr/src:/work/pi_mzink_umass_edu/SPRITE:/work/pi_mzink_umass_edu/SPRITE/nowcastnet/src:/work/pi_mzink_umass_edu/SPRITE/smaat_unet/src:/work/pi_mzink_umass_edu/SPRITE/casa_datatools/src

# Change to the appropriate directory
cd /work/pi_mzink_umass_edu/SPRITE/nowcastnet/

# Run the Python script
if [ -z "${checkpoint_path}" ]; then
    srun python src/nowcastnet/trainer.py \
            --model_name=${model}\
            --output_dir=\$output_dir\
            --job_id=\${SLURM_JOB_ID}\
            --strategy=${strategy}\
            --gpus=${gpus}\
            --num_nodes=${num_nodes}\
            --epochs=${epochs}\
            --batch_size=${batch_size}\
            --precision=${precision}\
            # --fast_dev_run
else
    srun python src/nowcastnet/trainer.py \
            --model_name=${model}\
            --output_dir=\$output_dir\
            --job_id=\${SLURM_JOB_ID}\
            --checkpoint_path=${checkpoint_path}\
            --strategy=${strategy}\
            --gpus=${gpus}\
            --num_nodes=${num_nodes}\
            --epochs=${epochs}\
            --batch_size=${batch_size}\
            --precision=${precision}\
            # --fast_dev_run
fi

# Deactivate the virtual environment
deactivate
EOT
