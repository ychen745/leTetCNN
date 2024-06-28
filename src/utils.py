import os
from scipy.io import loadmat
from torch_geometric import utils
from torch_geometric import transforms
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader
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


####LBO dataloader
def load_data(ftet, flbo, fmass, label, landmark=False, flmk=None):
    t2e = TetraToEdge()
    tet = tet_mesh.load_mesh(ftet)
    data = from_pymesh(tet)

    lbo = load_LBO(flbo, 'L').astype(np.float32)
    mass = load_LBO(fmass, 'm').astype(np.float32).todense()

    if landmark:
        lmk = np.loadtxt(flmk)
        lmk -= 1

    lambda_max = eigs(lbo, k=1, which='LM', return_eigenvectors=False)
    data = t2e(data)

    norm_pos = data.pos - torch.mean(data.pos, axis=0)
    norm_pos /= torch.max(torch.linalg.norm(norm_pos, axis=1))

    # disp_field = torch.randn((norm_pos.shape))
    # disp_field = disp_field * torch.min(torch.linalg.norm(norm_pos, axis=1)) / 10.0
    # norm_pos += disp_field

    if landmark:
        lmk_idx = torch.zeros(norm_pos.shape[0])
        lmk_idx[lmk] = 1

    edge_index_L, edge_weight_L = from_scipy_sparse_matrix(lbo)

    LBO_index2, LBO_weight2, LBO_index2_loop, LBO_weight2_loop =  segregate_self_loops(edge_index_L, edge_weight_L)

    LBO_index = torch.cat((LBO_index2, LBO_index2_loop),dim = 1)
    LBO_weight = torch.cat((-torch.abs(LBO_weight2),torch.abs(LBO_weight2_loop)))
    LBO_index[[0,1]] =  LBO_index[[1,0]]
    data.lambda_max = float(lambda_max.real)

    if landmark:
        data = Data(x = norm_pos, edge_index = data.edge_index, y = torch.tensor(label), pos = norm_pos,  LBO_index = LBO_index, LBO_weight = LBO_weight, mass = torch.tensor(mass), lmax = data.lambda_max, lmk = lmk_idx)
    else:
        data = Data(x = norm_pos, edge_index = data.edge_index, y = torch.tensor(label), pos = norm_pos,  LBO_index = LBO_index, LBO_weight = LBO_weight, mass = torch.tensor(mass), lmax = data.lambda_max)

    return data


if __name__ == '__main__':
    folder = '/data/hohokam/Yanxi/Data/tetCNN/328/lh/ad'

    tetlist = list_files(folder)
    lbolist = read_LBO(folder)
    masslist = read_mass(folder)
    cotlist = read_cot(folder)
    
    print(len(tetlist))
    print(len(lbolist))
    print(len(masslist))
    print(len(cotlist))