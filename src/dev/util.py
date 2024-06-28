import os
from scipy.io import loadmat
from torch_geometric import utils
from torch_geometric import transforms
import torch
from torch_geometric.data import Data, HeteroData, DataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from scipy.sparse.linalg import eigs
from torch_geometric.utils import get_laplacian
from torch_geometric.utils import segregate_self_loops,add_remaining_self_loops
from torch_geometric.transforms import TetraToEdge
import torch_geometric.transforms as T
from TetDataset import tetData
import pickle
# import random

def list_files(folder):
    flist = []
    for subfolder in os.listdir(folder):
        files = os.listdir(os.path.join(folder, subfolder))
        if len(files) > 0:
            for f in files:
                if f.endswith('.node'):
                    flist.append(os.path.join(folder, subfolder, f))
    return flist

def read_LBO(folder):
    flist = []
    for subfolder in os.listdir(folder):
        files = os.listdir(os.path.join(folder, subfolder))
        if len(files) > 0:
            if 'LBO.mat' in os.listdir(os.path.join(folder, subfolder)):
                flist.append(os.path.join(folder, subfolder, 'LBO.mat'))
    return flist

def read_mass(folder):
    flist = []
    for subfolder in os.listdir(folder):
        files = os.listdir(os.path.join(folder, subfolder))
        if len(files) > 0:
            if 'mass.mat' in os.listdir(os.path.join(folder, subfolder)):
                flist.append(os.path.join(folder, subfolder, 'mass.mat'))
    return flist

def read_cot(folder):
    flist = []
    for subfolder in os.listdir(folder):
        files = os.listdir(os.path.join(folder, subfolder))
        if len(files) > 0:
            if 'cot.mat' in os.listdir(os.path.join(folder, subfolder)):
                flist.append(os.path.join(folder, subfolder, 'cot.mat'))
    return flist




def list_files_lr(lfolder, rfolder):
    flist = []
    for subfolder in os.listdir(lfolder):
        if subfolder in os.listdir(rfolder):
            files = os.listdir(os.path.join(lfolder, subfolder))
            if len(files) > 0:
                for lf in files:
                    if lf.endswith('.node'):
                        for rf in os.listdir(os.path.join(rfolder, subfolder)):
                            if rf.endswith('.node'):
                                flist.append((os.path.join(lfolder, subfolder, lf), os.path.join(rfolder, subfolder, rf)))
    return flist


def read_LBO_lr(lfolder, rfolder):
    flist = []
    for subfolder in os.listdir(lfolder):
        if subfolder in os.listdir(rfolder):
            files = os.listdir(os.path.join(lfolder, subfolder))
            if len(files) > 0:
                if 'LBO.mat' in os.listdir(os.path.join(lfolder, subfolder)) and 'LBO.mat' in os.listdir(os.path.join(rfolder, subfolder)):
                    flist.append((os.path.join(lfolder, subfolder, 'LBO.mat'), os.path.join(rfolder, subfolder, 'LBO.mat')))
    return flist

def read_mass_lr(lfolder, rfolder):
    flist = []
    for subfolder in os.listdir(lfolder):
        if subfolder in os.listdir(rfolder):
            files = os.listdir(os.path.join(lfolder, subfolder))
            if len(files) > 0:
                if 'mass.mat' in os.listdir(os.path.join(lfolder, subfolder)) and 'mass.mat' in os.listdir(os.path.join(rfolder, subfolder)):
                    flist.append((os.path.join(lfolder, subfolder, 'mass.mat'), os.path.join(rfolder, subfolder, 'mass.mat')))
    return flist

def read_cot_lr(lfolder, rfolder):
    flist = []
    for subfolder in os.listdir(lfolder):
        if subfolder in os.listdir(rfolder):
            files = os.listdir(os.path.join(lfolder, subfolder))
            if len(files) > 0:
                if 'cot.mat' in os.listdir(os.path.join(lfolder, subfolder)) and 'cot.mat' in os.listdir(os.path.join(rfolder, subfolder)):
                    flist.append((os.path.join(lfolder, subfolder, 'cot.mat'), os.path.join(rfolder, subfolder, 'cot.mat')))
    return flist

def read_wks_lr(lfolder, rfolder):
    flist = []
    for subfolder in os.listdir(lfolder):
        if subfolder in os.listdir(rfolder):
            files = os.listdir(os.path.join(lfolder, subfolder))
            if len(files) > 0:
                if 'WKS.txt' in os.listdir(os.path.join(lfolder, subfolder)) and 'WKS.txt' in os.listdir(os.path.join(rfolder, subfolder)):
                    flist.append((os.path.join(lfolder, subfolder, 'WKS.txt'), os.path.join(rfolder, subfolder, 'WKS.txt')))
    return flist



def load_LBO(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containing the field name (default='shape')
    """
    
    return loadmat(path_file)[name_field]



def load_data(datafile):
    try:
        with open(datafile,"rb") as f:
            return pickle.load(f)
    except:
        x = []
    

def save_data(data, datafile):
    with open(datafile, "wb") as f:
        pickle.dump(data, f)


####LBO dataloader
def tet_data_LBO(list_tet, list_LBO, list_mass, positive):
    list_tet_data = []
    counter = 0
    l = len(list_tet)
    
    for i in range (l):
        t2e = TetraToEdge()
        data = utils.convert.from_pymesh(list_tet[i])
        L = list_LBO[i]
        m = list_mass[i].todense()
        lambda_max = eigs(L, k=1, which='LM', return_eigenvectors=False)
        data = t2e(data)
        norm_tet = data.pos - torch.mean(data.pos, axis=0)
        norm_tet /= torch.max(torch.linalg.norm(norm_tet, axis=1))
        edge_index_L, edge_weight_L = from_scipy_sparse_matrix(L)
        LBO_index2, LBO_weight2, LBO_index2_loop, LBO_weight2_loop =  segregate_self_loops(edge_index_L, edge_weight_L)

        LBO_index = torch.cat((LBO_index2, LBO_index2_loop),dim = 1)
        LBO_weight = torch.cat((-torch.abs(LBO_weight2),torch.abs(LBO_weight2_loop)))
        LBO_index[[0,1]] =  LBO_index[[1,0]]
        data.lambda_max = float(lambda_max.real)

        if positive:
            data = Data(x = norm_tet,edge_index = data.edge_index,y = torch.tensor(1), pos = norm_tet,  LBO_index = LBO_index, LBO_weight = LBO_weight, mass = torch.tensor(m),lmax = data.lambda_max )
        else:
            data = Data(x = norm_tet,edge_index = data.edge_index,y = torch.tensor(0), pos = norm_tet,  LBO_index = LBO_index, LBO_weight = LBO_weight, mass = torch.tensor(m),lmax = data.lambda_max )

        #data = Data(x = torch.from_numpy(AD_list_WKS[counter]).to(torch.float),edge_index = data.edge_index,y = torch.tensor(1), pos = data.pos )
        
        list_tet_data.append((data))

    return list_tet_data



if __name__ == '__main__':
    lfolder = '/scratch/ychen855/tetCNN/data/328/lh/mci'
    rfolder = '/scratch/ychen855/tetCNN/data/328/rh/mci'
    tetlist = list_lr(lfolder, rfolder)
    lbolist = list_lr(lfolder, rfolder)
    masslist = list_lr(lfolder, rfolder)
    cotlist = list_lr(lfolder, rfolder)
    print(len(tetlist))
    print(len(lbolist))
    print(len(masslist))
    print(len(cotlist))