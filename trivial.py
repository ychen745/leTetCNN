import os
import numpy as np
import pandas as pd
import shutil

folder = '/data/hohokam/Yanxi/Data/tetCNN/328/lh/ad'
flist = []
for pid in os.listdir(folder):
    if '_' not in pid:
        flist.append(pid)

root = '/data/amciilab/processedDataset/ADNI/ADNI-FS'
attr_list = ['Hippocampal_tail',
    'subiculum-body',
    'CA1-body',
    'subiculum-head',
    'hippocampal-fissure',
    'presubiculum-head',
    'CA1-head',
    'presubiculum-body',
    'parasubiculum',
    'molecular_layer_HP-head',
    'molecular_layer_HP-body',
    'GC-ML-DG-head',
    'CA3-body',
    'GC-ML-DG-body',
    'CA4-head',
    'CA4-body',
    'fimbria',
    'CA3-head',
    'HATA',
    'Whole_hippocampal_body',
    'Whole_hippocampal_head',
    'Whole_hippocampus']

sample_dict = dict()

for pid in flist:
    with open(os.path.join(root, pid, 'mri', 'lh.hippoSfVolumes-T1.v22.txt')) as f_info:
        for line in f_info:
            attr_dict = dict()
            linelist = line[:-1].split() if line[-1] == '\n' else line
            attr_dict[linelist[0]] = linelist[1]
            sample_dict[pid] = attr_dict

with open('hippo_info_ad.csv', 'w') as fout:
    fout.write(','.join(attr_list) + '\n')
    for pid in sample_dict.keys():
        line = []
        for attr in attr_list:
            line.append(attr)
        fout.write(','.join(line) + '\n')




            
                
