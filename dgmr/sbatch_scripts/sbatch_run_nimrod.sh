#!/bin/bash

#=========================#
#    User Configuration   #
#=========================#

# Define variables
num_nodes=2
gpus_per_node=4
partition="uri-gpu"
constraint="a100-80g"
time_limit="12:00:00"
project="balancing"
job_name="NIMROD"
output_base="/work/pi_mzink_umass_edu/SPRITE/skillful_nowcasting/output/${project}"

# Derived variables
job_name_full="${job_name}-${num_nodes}N-${gpus_per_node}G"

sbatch <<EOT
#!/bin/bash

#=========================#
#     SLURM Directives    #
#=========================#

#SBATCH --job-name=${job_name_full}-%j
#SBATCH --partition=${partition}
#SBATCH --constraint=${constraint}
#SBATCH --nodes=${num_nodes}
#SBATCH --gres=gpu:${gpus_per_node}
#SBATCH --ntasks-per-node=${gpus_per_node}
#SBATCH --time=${time_limit}
#SBATCH --output=${output_base}/${job_name_full}-%j/log-%j.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=500G # Helps with OOM

#=========================#
#      Job Execution      #
#=========================#

# Load the necessary module
module load conda/latest

# Create the output directory after SLURM_JOB_ID is available
OUTPUT_DIR="${output_base}/${job_name_full}-\${SLURM_JOB_ID}"
mkdir -p "\${OUTPUT_DIR}"

# Print job and directory details
echo "Job ID: \${SLURM_JOB_ID}"
echo "Output directory: \${OUTPUT_DIR}"

# Check for available Infiniband interfaces and set NCCL_SOCKET_IFNAME dynamically
available_ib_interface=\$(ip -o link show | grep 'state UP' | grep -E 'ib[0-9]|ibs[0-9]' | awk -F: '{print \$2}' | head -n 1 | tr -d ' ')

if [ -z "\$available_ib_interface" ]; then
  echo "Error: No active Infiniband interface found. Cancelling job."
  exit 1
else
  echo "Using Infiniband interface: \$available_ib_interface"
  export NCCL_SOCKET_IFNAME=\$available_ib_interface
fi

# Print the NCCL settings
echo "NCCL_SOCKET_IFNAME: \$NCCL_SOCKET_IFNAME"

# Set additional NCCL settings for Infiniband communication
export NCCL_IB_DISABLE=0               # Enable Infiniband
export NCCL_NET_GDR_LEVEL=0            # Adjust based on your setup
export NCCL_P2P_LEVEL=SYS              # Use system-level P2P communication

# Set the master address using the first node in the SLURM_NODELIST
MASTER_HOST=\$(scontrol show hostnames \$SLURM_NODELIST | head -n 1)

# Get the IP address associated with the Infiniband interface on the master host
MASTER_IP=\$(srun --nodes=1 --ntasks=1 --exclusive -w \$MASTER_HOST ip -o -4 addr show \$NCCL_SOCKET_IFNAME | awk '{print \$4}' | cut -d/ -f1)

# Set environment variables for DDP
export MASTER_ADDR=\$MASTER_IP
export MASTER_PORT=12345

# Print the MASTER_HOST and MASTER_IP for verification
echo "MASTER_HOST: \$MASTER_HOST"
echo "MASTER_IP (\$NCCL_SOCKET_IFNAME): \$MASTER_IP"
echo "MASTER_PORT: \$MASTER_PORT"

# Activate the conda environment
conda activate dgmr-casa-venv

# Change to the appropriate directory
cd /work/pi_mzink_umass_edu/SPRITE/skillful_nowcasting

echo "Executing Python Script"

# Set TensorFlow log level to ignore INFO and WARNING messages and  Disable oneDNN optimizations if not needed
export TF_CPP_MIN_LOG_LEVEL=3  # Set to 2 to show only errors, 3 to show only critical messages
export TF_ENABLE_ONEDNN_OPTS=0

# Run the Python script
srun python3 train/run_nimrod_updated.py \
  --nodes ${num_nodes} \
  --gpus ${gpus_per_node} \
  --partition ${partition} \
  --job_id \${SLURM_JOB_ID} \
  --dirpath "\${OUTPUT_DIR}" 

echo "Finished running"

# Deactivate the conda environment
conda deactivate
EOT