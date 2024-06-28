import os
import torch
from torch_geometric.data import Data, Dataset
import utils


class TetDataset(Dataset):
    def __init__(self, pos_list, neg_list, landmark=False, n_lmk=None, bnn=None):
        super().__init__()
        self.samples = pos_list + neg_list
        self.labels = [1] * len(pos_list) + [0] * len(neg_list)
        self.landmark = landmark
        self.n_lmk = n_lmk
        self.bnn = bnn
    
    def len(self):
        return len(self.samples)
    
    def get(self, idx):
        ftet = os.path.join(self.samples[idx], 'lh_hippo.1.node')
        flbo = os.path.join(self.samples[idx], 'cot.mat')
        fmass = os.path.join(self.samples[idx], 'mass.mat')
        if self.landmark:
            flmk = os.path.join(self.samples[idx], 'landmark_' + str(self.n_lmk) + '_' + str(self.bnn) + '.lmk')
        else:
            flmk = None

        return utils.load_data(ftet, flbo, fmass, self.labels[idx], self.landmark, flmk)