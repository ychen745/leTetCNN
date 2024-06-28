import os
import numpy as np
import warnings
import pickle
import random
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class PointDataSet(Dataset):
    def __init__(self, pos_folder, neg_folder, group, training=True, n_cv=5):
        self.pos_list = [os.path.join(pos_folder, pid) for pid in os.listdir(pos_folder)]
        self.neg_list = [os.path.join(neg_folder, pid) for pid in os.listdir(neg_folder)]
        self.group = group
        self.f_list = self.pos_list + self.neg_list
        self.labels = [1] * len(self.pos_list) + [0] * len(self.neg_list)
        self.data_list = [t for t in zip(self.f_list, self.labels)]
        random.seed(42)
        random.shuffle(self.data_list)
        if training:
            if group == n_cv - 1:
                self.use_list = self.data_list[:len(self.data_list)*group//n_cv]
            else:
                self.use_list = self.data_list[:len(self.data_list)*group//n_cv] + self.data_list[len(self.data_list)*(group+1)//n_cv:]
        else:
            self.use_list = self.data_list[len(self.data_list)*group//n_cv:len(self.data_list)*(group+1)//n_cv]
            

    def __len__(self):
        return len(self.use_list)

    def __getitem__(self, index):
        node_file = os.path.join(self.use_list[index][0], 'lh_hippo.1.node')
        node_mtx = np.loadtxt(node_file)[1:,1:]
        # print(node_mtx.shape)
        # node_mtx.astype(np.float32)

        return torch.from_numpy(node_mtx).float(), self.use_list[index][1]


if __name__ == '__main__':
    pos_folder = '/data/hohokam/Yanxi/Data/tetCNN/328/lh/ad'
    neg_folder = '/data/hohokam/Yanxi/Data/tetCNN/328/lh/cn'
    ds = PointDataset(pos_folder, neg_folder, 0)
    dl = DataLoader(ds, shuffle=False)
    for data in dl:
        print(data[0].shape, data[1])
