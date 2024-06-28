import os
import shutil
import random

lst = 'sorted_cn.txt'
src = '/data/hohokam/Yanxi/Data/tetCNN/328/lh/cn_2'
dst = '/data/hohokam/Yanxi/Data/tetCNN/328/lh/cn'
n = 214

idx = 0
with open(lst) as f:
    for line in f:
        pid = line[:-1] if line[-1] == '\n' else line
        if (pid in os.listdir(src)) and (pid not in os.listdir(dst)):
            idx += 1
            # print(idx)
            shutil.move(os.path.join(src, pid), os.path.join(dst, pid))
        if idx == n:
            break