#!/bin/bash

#SBATCH -N 1                        # number of compute nodes
#SBATCH -n 4                        # number of cores
#SBATCH -p general                      # Use gpu partition
#SBATCH --mem=128G
#SBATCH -G a100:1
#SBATCH -t 1-00:00:00                  # wall time (D-HH:MM)
#SBATCH -o /data/hohokam/Yanxi/Code/leTetCNN/scripts_yc/job_logs/ad_mci_tetcnn_0.out
#SBATCH -e /data/hohokam/Yanxi/Code/leTetCNN/scripts_yc/job_logs/ad_mci_tetcnn_0.err
#SBATCH --mail-type=END             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=ychen855@asu.edu # send-to address

# module purge    # Always purge modules to ensure a consistent environment
# module load cuda-12.1.1-gcc-12.1.0
# source ~/.bashrc
# conda activate cu121

POS=/data/ad
NEG=/data/mci
LMK=200
BNN=100
LR=0.001
WD=0.0001
NETWORK=gnn
N_EPOCH=80
N_WORKERS=8
FOLD=0
N_FOLD=5
MAXLR=0.005
SAVE_FREQ=20
VAL_FREQ=1
CHECKPOINT_DIR=/leTetCNN/checkpoints/ad_mci

HOST_DATA=/data/hohokam/Yanxi/Data/tetCNN/328/lh
VM_DATA=/data
HOST_CODE=/data/hohokam/Yanxi/Code/leTetCNN
VM_CODE=/leTetCNN
IMAGE=/data/hohokam/Yanxi/Code/tetcnn.sif
apptainer exec --bind ${HOST_DATA}:${VM_DATA},${HOST_CODE}:${VM_CODE} ${IMAGE} python3 ${VM_CODE}/src/tetCNN_lm.py --pos ${POS} --neg ${NEG} --n_workers ${N_WORKERS} --maxlr ${MAXLR} --val_freq ${VAL_FREQ} --save_freq ${SAVE_FREQ} --checkpoint_dir ${CHECKPOINT_DIR} --n_epoch ${N_EPOCH} --lr ${LR} --wd ${WD} --fold ${FOLD} --n_fold ${N_FOLD} --load
