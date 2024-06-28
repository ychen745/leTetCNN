import os

n_lmk = 200
k = 100
landmark = False

# root = '/data/hohokam/Yanxi/Data/tetCNN/328/lh'
# exp_pairs = [('ad', 'cn'), ('ad', 'mci'), ('mci', 'cn')]
# lr_wd_dict = {('ad', 'cn'): ('0.0002', '0.00002'), ('ad', 'mci'): ('0.0001', '0.00001'), ('mci', 'cn'): ('0.0001', '0.00001')}

root = '/data/hohokam/Yanxi/Data/tetCNN/328/lh'
exp_pairs = [('ad', 'cn')]
lrs = [1e-3]
wds = [1e-4]
networks = ['gnn']

n_epoch = 80
save_freq = 20
val_freq = 1
checkpoint_root = '/scratch/czhu62/yc/leTetCNN/checkpoints'
for exp_pair in exp_pairs:
    if exp_pair[0] + '_' + exp_pair[1] not in os.listdir(checkpoint_root):
        os.mkdir(os.path.join(checkpoint_root, exp_pair[0] + '_' + exp_pair[1]))
maxlr = 5e-3
n_workers = 8

for cv in range(1):
    for lr in lrs:
        for wd in wds:
            for network in networks:
                for pair in exp_pairs:
                    pos_neg = [(os.path.join(root, pair[0]), os.path.join(root, pair[1]))]
                    checkpoint_dir = os.path.join(checkpoint_root, pair[0] + '_' + pair[1])
                    for file_pair in pos_neg:
                        pos_folder = file_pair[0]
                        neg_folder = file_pair[1]
                        pos_name = pos_folder.split('/')[-1]
                        neg_name = neg_folder.split('/')[-1]
                        if landmark:
                            fout = open(os.path.join('sbatch_jobs', 'job_' + pos_name + '_' + neg_name + '_' + str(n_lmk) + '_' + str(k) + '_' + network + '_' + str(cv) + '.sh'), 'w')
                        else:
                            fout = open(os.path.join('sbatch_jobs', 'job_' + pos_name + '_' + neg_name + '_' + str(n_lmk) + '_' + str(k) + '_' + str(cv) + '_tetcnn.sh'), 'w')
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
                            out_lines.append('#SBATCH -o /scratch/czhu62/yc/leTetCNN/scripts/job_logs/' + str(pos_name) + '_' + str(neg_name) + '_' + str(n_lmk) + '_' + str(k) + '_' + network + '_' + str(cv) + '.out')
                            out_lines.append('#SBATCH -e /scratch/czhu62/yc/leTetCNN/scripts/job_logs/' + str(pos_name) + '_' + str(neg_name) + '_' + str(n_lmk) + '_' + str(k) + '_' + network + '_' + str(cv) + '.err')
                        else:
                            out_lines.append('#SBATCH -o /scratch/czhu62/yc/leTetCNN/scripts/job_logs/' + str(pos_name) + '_' + str(neg_name) + '_tetcnn' + '_' + str(cv) + '.out')
                            out_lines.append('#SBATCH -e /scratch/czhu62/yc/leTetCNN/scripts/job_logs/' + str(pos_name) + '_' + str(neg_name) + '_tetcnn' + '_' + str(cv) + '.err')
                        out_lines.append('#SBATCH --mail-type=END             # Send a notification when the job starts, stops, or fails')
                        out_lines.append('#SBATCH --mail-user=ychen855@asu.edu # send-to address\n')

                        out_lines.append('module purge    # Always purge modules to ensure a consistent environment')
                        out_lines.append('module load cuda-12.1.1-gcc-12.1.0')
                        out_lines.append('source ~/.bashrc')
                        out_lines.append('conda activate cu121\n')

                        out_lines.append('POS=' + pos_folder)
                        out_lines.append('NEG=' + neg_folder)
                        out_lines.append('LMK=' + str(n_lmk))
                        out_lines.append('K=' + str(k))
                        out_lines.append('LR=' + str(lr))
                        out_lines.append('NETWORK=' + network)
                        out_lines.append('N_EPOCH=' + str(n_epoch))
                        out_lines.append('N_WORKERS=' + str(n_workers))
                        out_lines.append('CV=' + str(cv))
                        out_lines.append('MAXLR=' + str(maxlr))
                        out_lines.append('SAVE_FREQ=' + str(save_freq))
                        out_lines.append('VAL_FREQ=' + str(val_freq))
                        out_lines.append('CHECKPOINT_DIR=' + checkpoint_dir + '\n')

                        if landmark:
                            out_lines.append('python /scratch/czhu62/yc/leTetCNN/src/tetCNN_lm.py --pos ${POS} --neg ${NEG} --n_workers ${N_WORKERS} --maxlr ${MAXLR} --val_freq ${VAL_FREQ} --save_freq ${SAVE_FREQ} --checkpoint_dir ${CHECKPOINT_DIR} --landmark --n_lmk ${LMK} --k ${K} --n_epoch ${N_EPOCH} --network ${NETWORK} --lr ${LR} --wd ${WD} --cv ${CV} --load')
                        else:
                            out_lines.append('python /scratch/czhu62/yc/leTetCNN/src/tetCNN_lm.py --pos ${POS} --neg ${NEG} --n_workers ${N_WORKERS} --maxlr ${MAXLR} --val_freq ${VAL_FREQ} --save_freq ${SAVE_FREQ} --checkpoint_dir ${CHECKPOINT_DIR} --n_epoch ${N_EPOCH} --lr ${LR} --wd ${WD} --cv ${CV} --load')

                        fout.write('\n'.join(out_lines) + '\n')
