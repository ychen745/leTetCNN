import torch
from torch.utils.data import Dataset

class tetData():
    def __init__(self, lh=None, rh=None, y=None):
        self.lh = lh
        self.rh = rh
        self.y = y

class PairDataset(Dataset):
    def __init__(self, ldata, rdata):
        self.ldata = data
        self.rdata = data
    
    def __len__(self):
        return len(self.ldata)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.ldata[idx], self.rdata[idx]