#!/bin/bash

#SBATCH -n 4                        # number of cores
#SBATCH -t 0-04:00:00                  # wall time (D-HH:MM:SS)
#SBATCH -o /scratch/ychen855/tetCNN/scripts/job_logs/slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e /scratch/ychen855/tetCNN/scripts/job_logs/slurm.%j.err             # STDERR (%j = JobId)
#SBATCH --mail-type=NONE             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=ychen855@asu.edu # send-to address

module purge    # Always purge modules to ensure a consistent environment

module load matlab/2022a

START_FOLDER="/scratch/ychen855/Plasma/meshes_pw/rh/fsaverage"
INCLUDE_FOLDER="/scratch/ychen855/tet_gen"
EXE_DIR="/scratch/ychen855/tet_gen"

cd /scratch/ychen855/tet_gen/matlab_compile
matlab -batch "generate_tet_lh $START_FOLDER $INCLUDE_FOLDER $EXE_DIR"
