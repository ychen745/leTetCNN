import os

root = '/data/hohokam/Yanxi/Data/plasma'
epoch = 100
# exp_pairs = [('ad', 'mci'), ('mci', 'cn'), ('ad', 'cn')]
exp_pairs = [('adni_pos', 'adni_neg'), ('bai_pos', 'bai_neg')]

for group in range(1):
    for pair in exp_pairs:
        pos_name = pair[0]
        neg_name = pair[1]
        # pos_neg = [(root + pair[0], root + pair[1])]
        pos_folder = os.path.join(root, pos_name)
        neg_folder = os.path.join(root, neg_name)

        fout = open(os.path.join('sbatch_pointnet', 'job_' + pos_name + '_' + neg_name + str(group) + '.sh'), 'w')
        out_lines = list()

        out_lines.append('#!/bin/bash\n')
        out_lines.append('#SBATCH -N 1                        # number of compute nodes')
        out_lines.append('#SBATCH -n 4                        # number of cores')
        out_lines.append('#SBATCH -p general                      # Use gpu partition')
        out_lines.append('#SBATCH --mem=128G')
        out_lines.append('#SBATCH -q public                 # Run job under wildfire QOS queue')
        out_lines.append('#SBATCH -G a100:1')
        out_lines.append('#SBATCH -t 0-06:00:00                  # wall time (D-HH:MM)')

        out_lines.append('#SBATCH -o /scratch/ychen855/tetCNN/scripts/job_logs_pointnet/' + str(pos_name) + '_' + str(neg_name) + '_' + str(group) + '.out')
        out_lines.append('#SBATCH -e /scratch/ychen855/tetCNN/scripts/job_logs_pointnet/' + str(pos_name) + '_' + str(neg_name) + '_' + str(group) + '.err')
        out_lines.append('#SBATCH --mail-type=END             # Send a notification when the job starts, stops, or fails')
        out_lines.append('#SBATCH --mail-user=ychen855@asu.edu # send-to address\n')

        out_lines.append('\n')
        out_lines.append('module purge    # Always purge modules to ensure a consistent environment')
        out_lines.append('module load cuda-11.8.0-gcc-12.1.0')
        out_lines.append('source ~/.bashrc')
        out_lines.append('conda activate tetcnn')

        out_lines.append('python /scratch/ychen855/tetCNN/src/pointnet_train.py --pos ' + pos_folder + ' --neg ' + neg_folder + ' --epoch ' + str(epoch) + ' --group ' + str(group))

        fout.write('\n'.join(out_lines) + '\n')
