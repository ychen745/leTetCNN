import os
import shutil
# import numpy as np
# import torch
# import math
# import random
# from scipy.io import loadmat

def compute_avg_dist():
    input = '/data/hohokam/Yanxi/Data/tetCNN/328/lh/ad/m027S0256L100606M515TCF/lh_hippo.1.node'
    with open(input) as f:
        n_node = int(f.readline().split()[0])
        coord = np.zeros((n_node, 3))
        for idx in range(n_node):
            x, y, z = f.readline().split()[1:4]
            coord[idx] = np.array([x, y, z])
        avg_coord = np.mean(coord, axis=0)
        max_coord = np.max(coord, axis=0)

        coord_t = torch.from_numpy(coord)

        norm_pos = coord_t - torch.mean(coord_t, axis=0)
        norm_pos /= torch.max(torch.linalg.norm(norm_pos, axis=1))

        print('final max: ', torch.max(norm_pos, axis=0).values)
        print('final min: ', torch.min(norm_pos, axis=0).values)
        print('final mean: ', torch.mean(norm_pos, axis=0))
        print('final std: ', torch.std(norm_pos, axis=0))

        disp_field = torch.randn((norm_pos.shape))
        disp_field = disp_field * torch.min(torch.linalg.norm(norm_pos, axis=1)) / 10.0
        print(torch.mean(disp_field, axis=0))
        print(torch.std(disp_field, axis=0))
        print('disp field: ', disp_field[:5])
        print('final pos: ', norm_pos[:5])

        norm_pos += disp_field
        print('final final pos: ', norm_pos[:5])


def check_tet(src, dst):
    flist = ['cot.mat', 'mass.mat']
    for pid in os.listdir(src):
        for f in flist:
            if f not in os.listdir(os.path.join(src, pid)):
                shutil.move(os.path.join(src, pid), os.path.join(dst, pid))
                break

def move_files(src, dst, lst):
    for pid in lst:
        if pid in os.listdir(src):
            shutil.move(os.path.join(src, pid), os.path.join(dst, pid))

def random_point_dropout(mtx, n_points=5000):
    idx_list = []
    i = 0
    while i < 5000:
        idx = math.floor(random.random() * mtx.shape[0])
        if idx not in idx_list:
            idx_list.append(idx)
            i += 1
    # print(len(idx_list))
    new_mtx = mtx[idx_list, :]

    return new_mtx


def subsample_node(src, dst, pid, n_points=2000):
    node_file = os.path.join(src, pid, 'lh_hippo.1.node')
    with open(node_file) as f:
        n_node = int(f.readline().split()[0])
        node_mtx = np.zeros((n_node, 3))
        for i in range(n_node):
            node_mtx[i] = f.readline().split()[1:4]
    node_mtx_new = random_point_dropout(node_mtx, n_points)
    # print(node_mtx_new.shape)
    if pid not in os.listdir(dst):
        os.mkdir(os.path.join(dst, pid))
    np.savetxt(os.path.join(dst, pid, 'lh_hippo.1.node'), node_mtx_new)


def reverse_sort(lst, lst_new):
    ret_line = []
    with open(lst) as f:
        for line in f:
            ret_line.insert(0, line[:-1] if line[-1] == '\n' else line)
    ret_lines = '\n'.join(ret_line) + '\n'
    with open(lst_new, 'w') as fout:
        fout.write(ret_lines)


if __name__ == '__main__':
    folder = '/data/hohokam/Yanxi/Data/tetCNN/328/lh/mci'
    n_adni = 0
    n_amcii = 0
    for f in os.listdir(folder):
        if '_' in f:
            n_adni += 1
        else:
            n_amcii += 1
    print(n_adni)
    print(n_amcii)
    # src = '/data/hohokam/Yanxi/Data/tetCNN/328/lh/mci'
    # dst = '/data/hohokam/Yanxi/Data/tetCNN/328/lh/mci_2'
    # lst = '/data/hohokam/Yanxi/Data/tetCNN/328/lh/mci_2.txt'

    # with open(lst) as f:
    #     for line in f:
    #         pid = line[:-1] if line[-1] == '\n' else line
    #         # print(pid)
    #         if pid in os.listdir(src):
    #             # print(pid)
    #             shutil.move(os.path.join(src, pid), os.path.join(dst, pid))

    # src = '/data/hohokam/Yanxi/Data/tetCNN/328/lh/cn'
    # dst = '/scratch/ychen855/tetCNN/data/cn'
    # for f in os.listdir(src):
    #     subsample_node(src, dst, f)

    # lst = 'sorted_cn.txt'
    # lst_new = 'sorted_cn_r.txt'
    # reverse_sort(lst, lst_new)