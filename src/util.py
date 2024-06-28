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
from torch_geometric.utils import to_undirected
import pickle
import numpy as np
from typing import Optional, Tuple
import tet_mesh
import random


class TetraToEdge(object):
    r"""Converts mesh tetras :obj:`[4, num_tetras]` to edge indices
    :obj:`[2, num_edges]` (functional name: :obj:`tetra_to_edge`).
    Args:
        remove_tetras (bool, optional): If set to :obj:`False`, the tetra tensor
            will not be removed.
    """

    def __init__(self, remove_tetras: bool = True):
        self.remove_tetras = remove_tetras

    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'tetra'):
            tetra = data.tetra
            edge_index = torch.cat([tetra[:2], tetra[1:3, :], tetra[-2:], tetra[::2], tetra[::3], tetra[1::2]], dim=1)
            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

            data.edge_index = edge_index
            if self.remove_tetras:
                data.tetra = None

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

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

def from_pymesh(mesh):
    r"""Converts a :pymesh file to a
    :class:`torch_geometric.data.Data` instance.
    Args:
        mesh (pymesh): A :obj:`pymesh` mesh.
    """

    pos = torch.from_numpy(mesh.vertices).to(torch.float)
    tetra = torch.from_numpy(mesh.voxels).to(torch.long).t().contiguous()

    return Data(pos=pos, tetra=tetra)


def load_LBO(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containing the field name (default='shape')
    """
    
    return loadmat(path_file)[name_field]


def load_dat(datafile):
    try:
        with open(datafile,"rb") as f:
            return pickle.load(f)
    except:
        x = []
    

def save_data(data, datafile):
    with open(datafile, "wb") as f:
        pickle.dump(data, f)


def load_data(pos_folders, neg_folders, num_workers, load, cv):
    ###loading AD files and LBOs
    print('begin loading data.')
    if load:   # if run python tetCNN_LBO.py --load it will go through loading all
        pos_tet_list = []
        pos_LBO_list = []
        pos_mass_list = []

        neg_tet_list = []
        neg_LBO_list = []
        neg_mass_list = []

        for i in range(len(pos_folders)):
            pos_folder = pos_folders[i]

            pos_files_tet = list_files(pos_folder)
            for sample in pos_files_tet:
                pos_tet_list.append(tet_mesh.load_mesh(sample))

            pos_files_LBO = read_cot(pos_folder)
            for sample in pos_files_LBO:
                pos_LBO_list.append(load_LBO(sample, 'L').astype(np.float32))
            
            pos_files_mass = read_mass(pos_folder)
            for sample in pos_files_mass:
                pos_mass_list.append(load_LBO(sample, 'm').astype(np.float32))

            ############### CTL or any other groups
        for i in range(len(neg_folders)):
            neg_folder = neg_folders[i]
            
            neg_files_tet = list_files(neg_folder)
            for sample in neg_files_tet:
                neg_tet_list.append(tet_mesh.load_mesh(sample))

            neg_files_LBO = read_cot(neg_folder)
            for sample in neg_files_LBO:
                neg_LBO_list.append(load_LBO(sample, 'L').astype(np.float32))
            
            neg_files_mass = read_mass(neg_folder)
            for sample in neg_files_mass:
                neg_mass_list.append(load_LBO(sample, 'm').astype(np.float32))

        pos_list_tet_data = tet_data_LBO(pos_tet_list, pos_LBO_list, pos_mass_list, True)
        neg_list_tet_data = tet_data_LBO(neg_tet_list, neg_LBO_list, neg_mass_list, False)
        ###combinig two together
        list_tet_data = pos_list_tet_data + neg_list_tet_data #+ MCI_list_tet_data
        # save_data(list_tet_data, 'tetmesh_198_left.dat')

    else:
        print('here')
        # list_tet_data = load_dat('tetmesh_198_left.dat', n_lmk, BNN)


    ###############
    torch.manual_seed(42)
    random.shuffle(list_tet_data)

    # print(len(list_tet_data))
    if cv == 5:
        train_dataset = list_tet_data[:len(list_tet_data)*cv//5]
    else:
        train_dataset = list_tet_data[:len(list_tet_data)*cv//5] + list_tet_data[len(list_tet_data)*(cv+1)//5:]
    test_dataset = list_tet_data[len(list_tet_data)*cv//5:len(list_tet_data)*(cv+1)//5]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=num_workers, pin_memory=True)

    print('finish loading.')

    return train_loader, test_loader

    ##############


####LBO dataloader
def tet_data_LBO(list_tet, list_LBO, list_mass, positive):
    list_tet_data = []
    counter = 0
    l = len(list_tet)
    
    for i in range (l):
        t2e = TetraToEdge()
        data = from_pymesh(list_tet[i])
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