#!/bin/bash

#=========================#
#    User Configuration   #
#=========================#

# Define variables
num_nodes=1
gpus_per_node=1
partition="uri-gpu"
constraint="a100"
time_limit="8:00:00"
memory="200G"

# Training settings
batch_size=32
epochs=3
project="TEST"
job_name="UNet"
output_base="/work/pi_mzink_umass_edu/SPRITE/outputs/${project}"
learning_rate=0.001

# Setup and configuration
model="SmaAt-UNet"
base_output_dir="/work/pi_mzink_umass_edu/SPRITE/outputs"
setup="N${num_nodes}-G${gpus_per_node}"
configuration="BS${batch_size}"
additional_configs="E${epochs}_LR${learning_rate}"
model_name="${model}-${additional_configs}"

sbatch <<EOT
#!/bin/bash

#=========================#
#     SLURM Directives    #
#=========================#

#SBATCH --job-name=${model_name}
#SBATCH --partition=${partition}
#SBATCH --constraint=${constraint}
#SBATCH --nodes=${num_nodes}
#SBATCH --gres=gpu:${gpus_per_node}
#SBATCH --ntasks-per-node=${gpus_per_node}
#SBATCH --time=${time_limit}
#SBATCH --output=${base_output_dir}/${project}/J%j-${model}-${setup}-${configuration}-${additional_configs}/log-J%j.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=${memory}

#=========================#
#      Job Execution      #
#=========================#

# Log GPU details
gpu_info=\$(nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader)
echo "Allocated GPU details:"
echo "\$gpu_info"
echo ""

#### Set Up Env ####
source /work/pi_mzink_umass_edu/venvs/uv-SPRITE/.venv-a/bin/activate

export PYTHONPATH=/work/pi_mzink_umass_edu/SPRITE/dgmr/src:/work/pi_mzink_umass_edu/SPRITE:/work/pi_mzink_umass_edu/SPRITE/nowcastnet/src:/work/pi_mzink_umass_edu/SPRITE/smaat_unet/src:/work/pi_mzink_umass_edu/SPRITE/casa_datatools/src

# Create the output directory after SLURM_JOB_ID is available
output_dir="${base_output_dir}/${project}/J\${SLURM_JOB_ID}-${model}-${setup}-${configuration}-${additional_configs}"
mkdir -p "\${output_dir}"

#### Execute script ####

# Change to the appropriate directory
cd /work/pi_mzink_umass_edu/SPRITE/smaat_unet

echo "output_dir: \$output_dir"

# Run the Python script
srun python src/smaat_unet/train/train_unet_casa_lightning.py \
    --accelerator gpu \
    --gpus ${gpus_per_node} \
    --num_nodes ${num_nodes} \
    --batch_size ${batch_size} \
    --learning_rate ${learning_rate} \
    --epochs ${epochs} \
    --strategy ddp \
    --output_path \${output_dir} 

# Deactivate the virtual environment
deactivate
EOT
