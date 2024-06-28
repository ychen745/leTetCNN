#!/bin/bash

#SBATCH -N 1                        # number of compute nodes
#SBATCH -n 4                        # number of cores
#SBATCH -p general                      # Use gpu partition
#SBATCH --mem=128G
#SBATCH -q public                 # Run job under wildfire QOS queue
#SBATCH -G a100:1
#SBATCH -t 0-04:00:00                  # wall time (D-HH:MM)
#SBATCH -o /scratch/ychen855/tetCNN/scripts/job_logs/slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e /scratch/ychen855/tetCNN/scripts/job_logs/slurm.%j.err             # STDERR (%j = JobId)
#SBATCH --mail-type=END             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=ychen855@asu.edu # send-to address
