#!/bin/bash

# GridCellLoss
# GPU
# Learning Rate

# ============ General Settings ============
num_nodes=3           # Number of nodes to use
gpus=4                # Number of GPUs per node
partition="uri-gpu"       # SLURM partition (e.g., gpu, uri-gpu, gpu-preempt)
project="GCR_LR" # Project name for outputs and logging
constraint="a100-80g,ib"   # GPU type

# ============ Training Settings ============
precision="32"             # Precision mode (e.g., "16-mixed", "32")
gen_steps=6                # Number of generation steps
batch_size=32              # Batch size for training
strategy="ddp_find_unused_parameters_true" # DDP strategy
checkpoint_path=""         # Checkpoint path to resume training
precip_weight_cap=24.0     # Precipitation weight cap (e.g., 1.0, 24.0, 64.0)
epochs=2000                 # Number of training epochs

# ============ Learning Rate and Weight Cap Combinations ============
declare -A combinations=(
    # Priority 1 combinations (will be submitted first)
    [1]="64.0 1e-5 5e-5"
    # [2]="128.0 1e-5 5e-5"
    # [3]="128.0 5e-5 2e-4"
    
    # Remaining combinations
    # [4]="24.0 1e-5 5e-5"
    # [5]="24.0 5e-5 2e-4"
    # [6]="24.0 1e-4 2e-4"
    # [7]="64.0 5e-5 2e-4"
    # [8]="64.0 1e-4 2e-4"
    # [9]="128.0 1e-4 2e-4"
)

# ============ Model Configuration ============
model="DGMR–CASA"
base_output_dir="/work/pi_mzink_umass_edu/SPRITE/outputs"
setup="N${num_nodes}-G${gpus}"
configuration="P${precision}_GS${gen_steps}_BS${batch_size}"

# Iterate through learning rate combinations and submit jobs
for combo in "${!combinations[@]}"; do
    IFS=" " read -r precip_weight_cap gen_lr disc_lr <<< "${combinations[$combo]}"
    additional_configs="PW${precip_weight_cap%.*}_E${epochs}_GLR${gen_lr}_DLR${disc_lr}"
    model_name="${model}-${additional_configs}"

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${model_name}
#SBATCH -p ${partition}
#SBATCH --constraint=${constraint}
#SBATCH -N ${num_nodes}
#SBATCH --gres=gpu:${gpus}
#SBATCH --ntasks-per-node=${gpus}
#SBATCH -t 100:00:00
#SBATCH -o ${base_output_dir}/${project}/J%j-${model}-${setup}-${configuration}-${additional_configs}/log-J%j.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=500G

# Job-specific output directory
output_dir="${base_output_dir}/${project}/J\${SLURM_JOB_ID}-${model}-${setup}-${configuration}-${additional_configs}"
mkdir -p "\$output_dir"

# Summary
echo "Job Configuration Summary:"
echo "--------------------------"
echo "Project: ${project}"
echo "Model: ${model}"
echo "Configuration: ${configuration}"
echo "Setup: ${setup}"
echo "Additional Configurations: ${additional_configs}"
echo "Output Directory: \$output_dir"
echo ""

# Load the necessary module
module load conda/latest

# Log GPU details
gpu_info=\$(nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader)
echo "Allocated GPU details:"
echo "\$gpu_info"
echo ""

# Activate the conda environment
conda activate dgmr-venv-v2

# Change to the appropriate directory
cd /work/pi_mzink_umass_edu/SPRITE/skillful_nowcasting

# Run the Python script
if [ -z "${checkpoint_path}" ]; then
    srun python3 train/run_casa_ddp_evalscore_GridCellLoss.py \
        --precision ${precision} \
        --gen_steps ${gen_steps} \
        --batch_size ${batch_size} \
        --output_dir "\$output_dir" \
        --model_name ${model_name} \
        --num_nodes ${num_nodes} \
        --gpus ${gpus} \
        --job_id \${SLURM_JOB_ID} \
        --partition ${partition} \
        --strategy ${strategy} \
        --precip_weight_cap ${precip_weight_cap} \
        --epochs ${epochs} \
        --gen_lr ${gen_lr} \
        --disc_lr ${disc_lr}
else
    srun python3 train/run_casa_ddp_evalscore_GridCellLoss.py \
        --precision ${precision} \
        --gen_steps ${gen_steps} \
        --batch_size ${batch_size} \
        --output_dir "\$output_dir" \
        --model_name ${model_name} \
        --num_nodes ${num_nodes} \
        --gpus ${gpus} \
        --checkpoint_path ${checkpoint_path} \
        --job_id \${SLURM_JOB_ID} \
        --partition ${partition} \
        --strategy ${strategy} \
        --precip_weight_cap ${precip_weight_cap} \
        --epochs ${epochs} \
        --gen_lr ${gen_lr} \
        --disc_lr ${disc_lr}
fi

# Deactivate the conda environment
conda deactivate
EOT
done
