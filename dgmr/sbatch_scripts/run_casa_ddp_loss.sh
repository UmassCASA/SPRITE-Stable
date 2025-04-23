#!/bin/bash

# ============ General Settings ============
num_nodes=1           # Number of nodes to use
gpus=1                # Number of GPUs per node
partition="uri-gpu,gpu"       # SLURM partition (e.g., gpu, uri-gpu, gpu-preempt)
project="TEST"
# project="Extended_Experiments_Precip_Weight_Cap_LR_GCR" # Project name for outputs and logging
# project="Extended_Experiments_Precip_Weight_Cap_LR_RecL" # Project name for outputs and logging
# project="Extended_Experiments_ExtendedPrecip_Weight_Cap_LR_GCR" # Project name for outputs and logging
# project="Extended_Experiments_ExtendedPrecip_Weight_Cap_LR_RecL" # Project name for outputs and logging


# constraint="a100-80g,ib"   # GPU type
constraint="l40s"

# ============ Training Settings ============
precision="32"             # Precision mode (e.g., "16-mixed", "32")
gen_steps=6                # Number of generation steps
batch_size=8              # Batch size for training
strategy="ddp_find_unused_parameters_true" # DDP strategy
checkpoint_path=""         # Checkpoint path to resume training
precip_weight_cap=24.0     # Precipitation weight cap (e.g., 1.0, 24.0, 64.0)
epochs=1                 # Number of training epochs
regularizer_type="grid_cell"
regularizer_lambda=20.0

# ============ Model Configuration ============
model="DGMR–CASA"
base_output_dir="/work/pi_mzink_umass_edu/SPRITE/outputs"
setup="N${num_nodes}-G${gpus}"
configuration="P${precision}_GS${gen_steps}_BS${batch_size}"

# ============ Learning Rate and Weight Cap Combinations ============
declare -A combinations=(
    # Grid Cell Loss combinations for different precipitation weight caps at 5e-5 gen_lr and 2e-4 disc_lr
    # project="Extended_Experiments_Precip_Weight_Cap_LR_GCR" # Project name for outputs and logging

    [1]="24.0 5e-5 2e-4 grid_cell 20.0"
    # [2]="32.0 5e-5 2e-4 grid_cell 20.0"
    # [3]="44.0 5e-5 2e-4 grid_cell 20.0"
    # [4]="56.0 5e-5 2e-4 grid_cell 20.0"
    # [5]="64.0 5e-5 2e-4 grid_cell 20.0"
    # [6]="80.0 5e-5 2e-4 grid_cell 20.0"
    # [7]="92.0 5e-5 2e-4 grid_cell 20.0"
    # [8]="104.0 5e-5 2e-4 grid_cell 20.0"
    # [9]="116.0 5e-5 2e-4 grid_cell 20.0"
    # [10]="128.0 5e-5 2e-4 grid_cell 20.0"

    # Grid Cell Loss combinations for different precipitation weight caps at 1e-5 gen_lr and 5e-5 disc_lr
    # [11]="24.0 1e-5 5e-5 grid_cell 20.0"
    # [12]="32.0 1e-5 5e-5 grid_cell 20.0"
    # [13]="44.0 1e-5 5e-5 grid_cell 20.0"
    # [14]="56.0 1e-5 5e-5 grid_cell 20.0"
    # [15]="64.0 1e-5 5e-5 grid_cell 20.0"
    # [16]="80.0 1e-5 5e-5 grid_cell 20.0"
    # [17]="92.0 1e-5 5e-5 grid_cell 20.0"
    # [18]="104.0 1e-5 5e-5 grid_cell 20.0"
    # [19]="116.0 1e-5 5e-5 grid_cell 20.0"
    # [20]="128.0 1e-5 5e-5 grid_cell 20.0"

    # Reconstruction Loss combinations for different precipitation weight caps at 5e-5 gen_lr and 2e-4 disc_lr
    # project="Extended_Experiments_Precip_Weight_Cap_LR_RecL" # Project name for outputs and logging

    # [21]="24.0 5e-5 2e-4 rec 0.01"
    # [22]="32.0 5e-5 2e-4 rec 0.01"
    # [23]="44.0 5e-5 2e-4 rec 0.01"
    # [24]="56.0 5e-5 2e-4 rec 0.01"
    # [25]="64.0 5e-5 2e-4 rec 0.01"
    # [26]="80.0 5e-5 2e-4 rec 0.01"
    # [27]="92.0 5e-5 2e-4 rec 0.01"
    # [28]="104.0 5e-5 2e-4 rec 0.01"
    # [29]="116.0 5e-5 2e-4 rec 0.01"
    # [30]="128.0 5e-5 2e-4 rec 0.01"

    # Reconstruction Loss combinations for different precipitation weight caps at 1e-5 gen_lr and 5e-5 disc_lr
    # [31]="24.0 1e-5 5e-5 rec 0.01"
    # [32]="32.0 1e-5 5e-5 rec 0.01"
    # [33]="44.0 1e-5 5e-5 rec 0.01"
    # [34]="56.0 1e-5 5e-5 rec 0.01"
    # [35]="64.0 1e-5 5e-5 rec 0.01"
    # [36]="80.0 1e-5 5e-5 rec 0.01"
    # [37]="92.0 1e-5 5e-5 rec 0.01"
    # [38]="104.0 1e-5 5e-5 rec 0.01"
    # [39]="116.0 1e-5 5e-5 rec 0.01"
    # [40]="128.0 1e-5 5e-5 rec 0.01"
    
    # Grid Cell Loss combinations for near 44th precipitation weight cap at 5e-5 gen_lr and 2e-4 disc_lr
    # project="Extended_Experiments_ExtendedPrecip_Weight_Cap_LR_GCR" # Project name for outputs and logging

    # [41]="36.0 5e-5 2e-4 grid_cell 20.0"
    # [42]="40.0 5e-5 2e-4 grid_cell 20.0"
    # 44
    # [43]="48.0 5e-5 2e-4 grid_cell 20.0"
    # [44]="52.0 5e-5 2e-4 grid_cell 20.0"
    # 56

    # Grid Cell Loss combinations for near 44th precipitation weight cap at 1e-5 gen_lr and 5e-5 disc_lr
    # [45]="36.0 1e-5 5e-5 grid_cell 20.0"
    # [46]="40.0 1e-5 5e-5 grid_cell 20.0"
    # 44
    # [47]="48.0 1e-5 5e-5 grid_cell 20.0"
    # [48]="52.0 1e-5 5e-5 grid_cell 20.0"
    # 56

    # Reconstruction Loss combinations for near 44th precipitation weight cap at 5e-5 gen_lr and 2e-4 disc_lr
    # project="Extended_Experiments_ExtendedPrecip_Weight_Cap_LR_RecL" # Project name for outputs and logging
    # [49]="36.0 5e-5 2e-4 rec 0.01"
    # [50]="40.0 5e-5 2e-4 rec 0.01"
    # 44
    # [51]="48.0 5e-5 2e-4 rec 0.01"
    # [52]="52.0 5e-5 2e-4 rec 0.01"
    # 56

    # Reconstruction Loss combinations for near 44th precipitation weight cap at 1e-5 gen_lr and 5e-5 disc_lr
    # [53]="36.0 1e-5 5e-5 rec 0.01"
    # [54]="40.0 1e-5 5e-5 rec 0.01"
    # 44
    # [55]="48.0 1e-5 5e-5 rec 0.01"
    # [56]="52.0 1e-5 5e-5 rec 0.01"
    # 56
    
    
)

# Iterate through learning combinations and submit jobs
for combo in "${!combinations[@]}"; do
    IFS=" " read -r precip_weight_cap gen_lr disc_lr regularizer_type regularizer_lambda <<< "${combinations[$combo]}"
    additional_configs="PW${precip_weight_cap%.*}_E${epochs}_GLR${gen_lr}_DLR${disc_lr}_${regularizer_type}"
    model_name="${model}-${additional_configs}"

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${project}
#SBATCH -p ${partition}
#SBATCH --constraint=${constraint}
#SBATCH -N ${num_nodes}
#SBATCH --gres=gpu:${gpus}
#SBATCH --ntasks-per-node=${gpus}
#SBATCH -t 47:00:00
#SBATCH -o ${base_output_dir}/${project}/J%j-${model}-${setup}-${configuration}-${additional_configs}/log-J%j.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=200G

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

# Log GPU details
gpu_info=\$(nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader)
echo "Allocated GPU details:"
echo "\$gpu_info"
echo ""

# Activate the virtual senvironment
source /work/pi_mzink_umass_edu/venvs/uv-SPRITE/.venv-a/bin/activate

export PYTHONPATH=/work/pi_mzink_umass_edu/SPRITE/dgmr/src:/work/pi_mzink_umass_edu/SPRITE:/work/pi_mzink_umass_edu/SPRITE/nowcastnet/code:/work/pi_mzink_umass_edu/SPRITE/smaat_unet/src:/work/pi_mzink_umass_edu/SPRITE/casa_datatools/src

# Change to the appropriate directory
cd /work/pi_mzink_umass_edu/SPRITE/dgmr


# Run the Python script
if [ -z "${checkpoint_path}" ]; then
    srun python src/dgmr/train/run_casa_dgmr.py \
        --project ${project} \
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
        --disc_lr ${disc_lr} \
        --regularizer_type ${regularizer_type} \
        --regularizer_lambda ${regularizer_lambda} \
        # --fast_dev_run
else
    srun python src/dgmr/train/run_casa_dgmr.py \
        --project ${project} \
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
        --disc_lr ${disc_lr} \
        --regularizer_type ${regularizer_type} \
        --regularizer_lambda ${regularizer_lambda}
fi

# Deactivate the virtual environment
deactivate
EOT
done
