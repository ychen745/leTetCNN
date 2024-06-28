import os
import shutil

# datasets = ['pos', 'neg']
data_folder = '/data/hohokam/Yanxi/Data/NACC_hippo'
matlab_folder = '/scratch/ychen855/tetCNN/src/matlab'

with open('job_template_matlab.sh') as f:
	lines = list()
	for line in f:
		if line[0] == '#':
			lines.append(line[:-1])

lines.append('module purge    # Always purge modules to ensure a consistent environment\n')
lines.append('module load matlab/2022a\n')

# for subfolder in os.listdir(data_folder):
# 	fout = open(os.path.join(tetgen_folder, 'sbatch_jobs_matlab', subfolder + '_' + half + '.sh'), 'w')
# 	out_lines = []
# 	start_folder = os.path.join(data_folder, subfolder)
# 	start_folder = start_folder if start_folder[-1] != '/' else start_folder[:-1]
# 	out_lines.append('cd ' + os.path.join(tetgen_folder, 'matlab') + '\n')
# 	out_lines.append('matlab -batch \"calc_LBO_lump_' + half + ' \"')
# 	fout.write('\n'.join(lines + out_lines))

idx = 0
for f in sorted(os.listdir(data_folder)):
	if idx >= 1000 and idx < 2000:
		if 'cot.mat' not in os.listdir(os.path.join(data_folder, f)) or 'mass.mat' not in os.listdir(os.path.join(data_folder, f)):
			with open(os.path.join('sbatch_jobs', 'job_LBO_' + f + '.sh'), 'w') as fout:
				out_lines = list()
				out_lines.append('INCLUDE_DIR=\"' + matlab_folder + '\"')
				out_lines.append('DATA_DIR=\"' + os.path.join(data_folder, f) + '\"\n')
				out_lines.append('cd ' + matlab_folder + '\n')
				out_lines.append('matlab -batch \"calc_LBO_lump_hippo ${INCLUDE_DIR} ${DATA_DIR}\"')
				fout.write('\n'.join(lines + out_lines))
	idx += 1

