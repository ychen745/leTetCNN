import os
import tet_mesh
import util
from scipy.io import loadmat
from torch_geometric import utils
from torch_geometric import transforms
import torch
from torch_geometric.data import Data
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from scipy.sparse.linalg import eigs
from torch_geometric.utils import get_laplacian
from torch_geometric.utils import segregate_self_loops,add_remaining_self_loops
import torch_geometric.transforms as T
import argparse
import random
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import model
import model_lm
# from model_lm import Net, saliency_map, grad_cam
from torch_geometric.data import Batch
from torch_geometric.nn import summary
from TetDataset import TetDataset


def train_(loader, model, scheduler, criterion):
    with torch.no_grad():
        CM=0
        correct = 0
        train_loss = 0
        n = 0
        for data in loader:
            data = data.to(device)
            out, _ = model(data=data, training=False)
            loss = criterion(out, data.y)
            train_loss += loss
            n += 1
            pred = out.max(1)[1]  # Use the class with highest probability.
            CM+=confusion_matrix(data.y.cpu(), pred.cpu(),labels=[0,1])
            correct += pred.eq(data.y).sum().item()
        return CM, train_loss / n

def train(loader, model, scheduler, criterion):
    correct1 = 0
    train_loss = 0
    n = 0
    for data in loader:
        data = data.to(device)
        out, _ = model(data=data, training=True)
        loss = criterion(out, data.y)
        train_loss += loss
        n += 1
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return train_loss / n

def test(loader, model, criterion):
    with torch.no_grad():
        CM=0
        correct = 0
        test_loss = 0
        n = 0
        for data in loader:
            data = data.to(device)
            out, _ = model(data=data, training=False)
            loss = criterion(out, data.y)
            test_loss += loss
            n += 1
            #print(h.edge_index)
    #         cams = cam_extractor(out.squeeze(0).argmax().item(), out)
            pred = out.max(1)[1]  # Use the class with highest probability.
            CM+=confusion_matrix(data.y.cpu(), pred.cpu(),labels=[0,1])
            correct += pred.eq(data.y).sum().item()
        return CM, out, test_loss / n


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--pos', nargs='+')
    # parser.add_argument('--neg', nargs='+')
    parser.add_argument('--pos', type=str)
    parser.add_argument('--neg', type=str)
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--n_lmk', type=int)
    parser.add_argument('--bnn', type=int)
    parser.add_argument('--n_epoch', type=int)
    parser.add_argument('--landmark', action='store_true')
    parser.add_argument('--network', type=str)
    parser.add_argument('--n_workers', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--wd', type=float)
    parser.add_argument('--n_fold', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--save_freq', type=int)
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--checkpoint_dir', type=str)
    parser.add_argument('--maxlr', type=float)

    args, _ = parser.parse_known_args()

    pos_list = [os.path.join(args.pos, ele) for ele in os.listdir(args.pos)]
    neg_list = [os.path.join(args.neg, ele) for ele in os.listdir(args.neg)]

    train_pos = pos_list[:len(pos_list)*args.fold//args.n_fold] + pos_list[len(pos_list)*(args.fold+1)//args.n_fold:]
    train_neg = neg_list[:len(neg_list)*args.fold//args.n_fold] + neg_list[len(neg_list)*(args.fold+1)//args.n_fold:]

    test_pos = pos_list[len(pos_list)*args.fold//args.n_fold:len(pos_list)*(args.fold+1)//args.n_fold]
    test_neg = neg_list[len(neg_list)*args.fold//args.n_fold:len(neg_list)*(args.fold+1)//args.n_fold]
    
    if not args.landmark:
        setattr(args, 'n_lmk', None)
        setattr(args, 'bnn', None)

    train_dataset = TetDataset(train_pos, train_neg, args.landmark, args.n_lmk, args.bnn)
    test_dataset = TetDataset(test_pos, test_neg, args.landmark, args.n_lmk, args.bnn)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=8, num_workers=args.n_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=8, num_workers=args.n_workers, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_lm.Net(args.landmark, args.network).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.maxlr, steps_per_epoch=len(train_loader), epochs=args.n_epoch)
    criterion = torch.nn.CrossEntropyLoss()

    res_test_acc = []

    for epoch in range(1, args.n_epoch+1):
        train_loss = train(train_loader, model, scheduler, criterion)
        if epoch % args.val_freq == 0:
            train_acc, val_loss = train_(train_loader, model, scheduler, criterion)
            test_acc, out, test_loss = test(test_loader, model, criterion)
        
        if epoch % args.save_freq == 0:
            torch.save(model, os.path.join(args.checkpoint_dir, str(epoch) + '.pth'))

        tn_train=train_acc[0][0]
        tp_train=train_acc[1][1]
        fp_train=train_acc[0][1]
        fn_train=train_acc[1][0]
        tr_acc = np.sum(np.diag(train_acc)/np.sum(train_acc))
        
        tn_test=test_acc[0][0]
        tp_test=test_acc[1][1]
        fp_test=test_acc[0][1]
        fn_test=test_acc[1][0]
        test_acc = np.sum(np.diag(test_acc)/np.sum(test_acc))
        test_sen=tp_test/(tp_test+fn_test)
        test_spe = tn_test/(tn_test+fp_test)
        test_prec=tp_test/(tp_test+fp_test)
        res_test_acc.append(test_acc)
        print('Epoch: {:02d}, Train loss: {:.4f}, Val loss: {:.4f}, Test loss: {:.4f}, Train acc: {:.4f}, Test acc: {:.4f}, Test sen: {:.4f}, Test spe: {:.4f}, Test prec: {:.4f}'.format(epoch, train_loss, val_loss, test_loss, tr_acc, test_acc,test_sen,test_spe,test_prec))

    print('best test acccuracy: ', max(res_test_acc))

###### gradcam ######


    # optimizer.zero_grad()
    # X = next(iter(test_loader))
    # X.to(device)
    # out,h = model(X)
    # model.final_conv_acts.retain_grad()
    # loss = criterion(out, X.y)  # Compute the loss.
    # #print(loss)
    # loss.backward()  # Derive gradients.

    # # saliency_map_weights = saliency_map()
    # grad_cam_weights = grad_cam(model.final_conv_acts, model.final_conv_grads)
    # scaled_grad_cam_weights = MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(grad_cam_weights).reshape(-1, 1))

    # from torch_geometric.nn import  knn_interpolate
    # tmp = torch.tensor(scaled_grad_cam_weights)
    # print(tmp.view(-1,1).cuda())
    # h.x = tmp.view(-1,1)
    # h = h.to(device)
    # print('dim h.x: ' + str(h.x.shape))
    # print('dim h.pos: ' + str(h.pos.shape))
    # print('dim x.pos: ' + str(X.pos.shape))
    # x = knn_interpolate(h.x, h.pos, X.pos)
    # # print('before savetxt')
    # # np.savetxt(os.path.join(os.path.abspath('.'), 'cams', 'gradcam_mci_cn.txt'),x.cpu(), fmt='%.6e')

