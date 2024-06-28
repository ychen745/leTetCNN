import os

n_fold = 5

n_lmk = 200
bnn = 100
landmark = False

root = '/data/hohokam/Yanxi/Data/tetCNN/328/lh'
exp_pairs = [('ad', 'cn')]
lrs = [1e-3]
wds = [1e-4]
networks = ['gnn']

n_epoch = 80
save_freq = 20
val_freq = 1
maxlr = 5e-3
n_workers = 8
checkpoint_root = '/scratch/czhu62/yc/leTetCNN/checkpoints'

host_data = root
vm_data = '/data'
host_code = '/scratch/czhu62/yc/leTetCNN'
vm_code = '/leTetCNN'
image = '/scratch/czhu62/yc/tetcnn.sif'

vm_checkpoint_root = os.path.join(vm_code, 'checkpoints')

for exp_pair in exp_pairs:
    if exp_pair[0] + '_' + exp_pair[1] not in os.listdir(checkpoint_root):
        os.mkdir(os.path.join(checkpoint_root, exp_pair[0] + '_' + exp_pair[1]))

for fold in range(1):
    for lr in lrs:
        for wd in wds:
            for network in networks:
                for pair in exp_pairs:
                    # pos_neg = [(os.path.join(root, pair[0]), os.path.join(root, pair[1]))]
                    pos_neg = [(os.path.join(vm_data, pair[0]), os.path.join(vm_data, pair[1]))]
                    # checkpoint_dir = os.path.join(checkpoint_root, pair[0] + '_' + pair[1])
                    checkpoint_dir = os.path.join(vm_checkpoint_root, pair[0] + '_' + pair[1])
                    for file_pair in pos_neg:
                        pos_folder = file_pair[0]
                        neg_folder = file_pair[1]
                        pos_name = pair[0]
                        neg_name = pair[1]
                        if landmark:
                            fout = open(os.path.join('sbatch_jobs', 'job_' + pos_name + '_' + neg_name + '_' + str(n_lmk) + '_' + str(k) + '_' + network + '_' + str(fold) + '.sh'), 'w')
                        else:
                            fout = open(os.path.join('sbatch_jobs', 'job_' + pos_name + '_' + neg_name + '_' + str(fold) + '_tetcnn.sh'), 'w')
                        out_lines = list()

                        out_lines.append('#!/bin/bash\n')
                        out_lines.append('#SBATCH -N 1                        # number of compute nodes')
                        out_lines.append('#SBATCH -n 8                        # number of cores')
                        out_lines.append('#SBATCH -p general                      # Use gpu partition')
                        out_lines.append('#SBATCH --mem=128G')
                        out_lines.append('#SBATCH -q grp_rwang133')
                        out_lines.append('#SBATCH -G h100:1')
                        out_lines.append('#SBATCH -t 1-00:00:00                  # wall time (D-HH:MM)')

                        if landmark:
                            out_lines.append('#SBATCH -o /scratch/czhu62/yc/leTetCNN/scripts/job_logs/' + str(pos_name) + '_' + str(neg_name) + '_' + str(n_lmk) + '_' + str(k) + '_' + network + '_' + str(fold) + '.out')
                            out_lines.append('#SBATCH -e /scratch/czhu62/yc/leTetCNN/scripts/job_logs/' + str(pos_name) + '_' + str(neg_name) + '_' + str(n_lmk) + '_' + str(k) + '_' + network + '_' + str(fold) + '.err')
                        else:
                            out_lines.append('#SBATCH -o /scratch/czhu62/yc/leTetCNN/scripts/job_logs/' + str(pos_name) + '_' + str(neg_name) + '_tetcnn' + '_' + str(fold) + '.out')
                            out_lines.append('#SBATCH -e /scratch/czhu62/yc/leTetCNN/scripts/job_logs/' + str(pos_name) + '_' + str(neg_name) + '_tetcnn' + '_' + str(fold) + '.err')
                        out_lines.append('#SBATCH --mail-type=END             # Send a notification when the job starts, stops, or fails')
                        out_lines.append('#SBATCH --mail-user=ychen855@asu.edu # send-to address\n')

                        out_lines.append('module purge    # Always purge modules to ensure a consistent environment')
                        out_lines.append('module load cuda-12.1.1-gcc-12.1.0')
                        out_lines.append('source ~/.bashrc')
                        out_lines.append('conda activate cu121\n')

                        out_lines.append('POS=' + pos_folder)
                        out_lines.append('NEG=' + neg_folder)
                        out_lines.append('LMK=' + str(n_lmk))
                        out_lines.append('BNN=' + str(bnn))
                        out_lines.append('LR=' + str(lr))
                        out_lines.append('WD=' + str(wd))
                        out_lines.append('NETWORK=' + network)
                        out_lines.append('N_EPOCH=' + str(n_epoch))
                        out_lines.append('N_WORKERS=' + str(n_workers))
                        out_lines.append('FOLD=' + str(fold))
                        out_lines.append('N_FOLD=' + str(n_fold))
                        out_lines.append('MAXLR=' + str(maxlr))
                        out_lines.append('SAVE_FREQ=' + str(save_freq))
                        out_lines.append('VAL_FREQ=' + str(val_freq))
                        out_lines.append('CHECKPOINT_DIR=' + checkpoint_dir + '\n')

                        out_lines.append('HOST_DATA=' + host_data)
                        out_lines.append('VM_DATA=' + vm_data)
                        out_lines.append('HOST_CODE=' + host_code)
                        out_lines.append('VM_CODE=' + vm_code)
                        out_lines.append('IMAGE=' + image)


                        if landmark:
                            out_lines.append('python /scratch/czhu62/yc/leTetCNN/src/tetCNN_lm.py --pos ${POS} --neg ${NEG} --n_workers ${N_WORKERS} --maxlr ${MAXLR} --val_freq ${VAL_FREQ} --save_freq ${SAVE_FREQ} --checkpoint_dir ${CHECKPOINT_DIR} --landmark --n_lmk ${LMK} --bnn ${BNN} --n_epoch ${N_EPOCH} --network ${NETWORK} --lr ${LR} --wd ${WD} --fold ${FOLD} --n_fold ${N_FOLD} --load')
                        else:
                            out_lines.append('apptainer exec --bind ${HOST_DATA}:${VM_DATA},${HOST_CODE}:${VM_CODE} ${IMAGE} python3 ${VM_CODE}/src/leTetCNN.py --pos ${POS} --neg ${NEG} --n_workers ${N_WORKERS} --maxlr ${MAXLR} --val_freq ${VAL_FREQ} --save_freq ${SAVE_FREQ} --checkpoint_dir ${CHECKPOINT_DIR} --n_epoch ${N_EPOCH} --lr ${LR} --wd ${WD} --fold ${FOLD} --n_fold ${N_FOLD} --load')

                        fout.write('\n'.join(out_lines) + '\n')
