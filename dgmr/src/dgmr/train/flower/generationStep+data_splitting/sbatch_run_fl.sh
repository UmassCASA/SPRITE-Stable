#!/bin/bash

#SBATCH -t 48:00:00        # Job time limit
#SBATCH -o /work/pi_mzink_umass_edu/SPRITE/skillful_nowcasting/output/sbatch_logs/fl_training_logs/DGMR-fl-starter-%j.out    # %j will be replaced with the job ID
#SBATCH --mail-type=ALL    # Send a notification when the job starts, stops, or fails; 'ALL' adds job completion email to the notification

if [ -z "$1" ]; then
  echo "missing client number"
  exit 1
fi
CLIENT_COUNT=$1

#server_job_output=$(sbatch sbatch_run_fl_server.sh  $CLIENT_COUNT)
server_job_output=$(sbatch sbatch_run_fl_server.sh  1)
server_job_id=$(echo $server_job_output | awk '{print $4}')

echo "$server_job_id: sbatch sbatch_run_fl_server.sh" >> jobs.rec


while true; do
  if [ -f "FL_Server_GPU.txt" ]; then
    echo "Find server domain: $(cat FL_Server_GPU.txt)"
    break
  else
    echo "Waiting for server domain..."
    sleep 5
  fi
done

for i in $(seq 0 $(($CLIENT_COUNT - 1))); do
  echo "Creating client $i..."
  client_job_output=$(sbatch sbatch_run_fl_client.sh $i)
  client_job_id=$(echo $client_job_output | awk '{print $4}')
  echo "$client_job_id: sbatch sbatch_run_fl_client.sh $i" >> jobs.rec
done


# Set the directory to monitor and the size threshold (in bytes)
#MONITOR_DIR="/home/zhexu_umass_edu/PycharmProjects/SPRITE/skillful_nowcasting/train/flower/generationStep+data_splitting/lightning_logs"
#SIZE_THRESHOLD=100000000  # 10GB
#
## Wait for the directory to appear
#while [ ! -d "$MONITOR_DIR" ]; do
#    echo "Waiting for directory $MONITOR_DIR to be created..."
#    sleep 5
#done
#
## Monitor the directory size continuously
#while true; do
#    # Calculate the size of the directory
#    DIR_SIZE=$(du -sb "$MONITOR_DIR" | awk '{print $1}')
#
#    # Check if the directory size exceeds the threshold
#    if [ "$DIR_SIZE" -gt "$SIZE_THRESHOLD" ]; then
#        echo "Directory size exceeded threshold. Clearing $MONITOR_DIR..."
#        rm -rf "$MONITOR_DIR"/*
#        echo "Directory cleared."
#    fi
#
#    # Wait before checking again
#    sleep 10
#done
