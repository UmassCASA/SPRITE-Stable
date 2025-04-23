#!/bin/bash

if [ -f "jobs.rec" ]; then
  while IFS= read -r line; do
    job_id=$(echo $line | awk -F: '{print $1}')
    command=$(echo $line | awk -F: '{print $2}')
    if [ -n "$job_id" ]; then
      echo "Canceling job: $job_id: $command"
      scancel $job_id
    fi
  done < jobs.rec

  > jobs.rec
  echo "All jobs are canceled, jobs records emptied"
else
  echo "jobs.rec not found"
fi

files_to_delete="FL_Server_GPU.txt"
if [ -f "$files_to_delete" ]; then
    echo "Deleting file $files_to_delete..."
    rm "$files_to_delete"
  else
    echo "File $files_to_delete not found, skip"
  fi

for dir in "./lightning_logs_"*; do
  if [ -d "$dir" ]; then
    echo "Deleting directory: $dir"
    rm -r "$dir"
  fi
done
