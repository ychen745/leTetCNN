import pointnet2_cls_ssg
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from PointDataset import PointDataSet
from sklearn.metrics import confusion_matrix

def train_(loader):
    #model.eval()
    with torch.no_grad():
        CM=0
        correct = 0
        train_loss = 0
        n = 0
        for data, label in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(device)
            data = data.transpose(2, 1)
            label = label.to(device)
            out, trans_feat = model(data)
            loss = criterion(out, label, trans_feat)
            train_loss += loss
            n += 1
            pred = out.max(1)[1]  # Use the class with highest probability.
            CM+=confusion_matrix(label.cpu(), pred.cpu(),labels=[0,1])
            correct += pred.eq(label).sum().item()
        return CM, train_loss / n  #correct / len(loader.dataset)  # Derive ratio of correct predictions.

def train(loader):
    # model.train()
    correct1 = 0
    train_loss = 0
    n = 0
    for data, label in loader:
        # data_list = data.to_data_list()
        data = data.to(device)
        data = data.transpose(2, 1)
        label = label.to(device)
        out, trans_feat = model(data)
        loss = criterion(out, label, trans_feat)
        train_loss += loss
        n += 1
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return train_loss / n

def test(loader):
    with torch.no_grad():
        CM=0
        correct = 0
        test_loss = 0
        n = 0
        for data, label in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(device)
            data = data.transpose(2, 1)
            label = label.to(device)
            out, trans_feat = model(data)
            loss = criterion(out, label, trans_feat)
            test_loss += loss
            n += 1
            #print(h.edge_index)
    #         cams = cam_extractor(out.squeeze(0).argmax().item(), out)
            pred = out.max(1)[1]  # Use the class with highest probability.
            CM+=confusion_matrix(label.cpu(), pred.cpu(),labels=[0,1])
            correct += pred.eq(label).sum().item()
        return CM, out, test_loss / n #correct / len(loader.dataset)  # Derive ratio of correct predictions.

if __name__ == '__main__':
    # load_data(['/tetCNN/data/test/lh/ad'], ['/tetCNN/data/test/rh/ad'], ['/tetCNN/data/test/lh/cn'], ['/tetCNN/data/test/rh/cn'], True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos', dest='pos')
    parser.add_argument('--neg', dest='neg')
    parser.add_argument('--epoch', type=int, dest='n_epoch')
    parser.add_argument('--lr', type=float, default=0.001, dest='lr')
    parser.add_argument('--wd', type=float, default=1e-4, dest='wd')
    parser.add_argument('--group', type=int, dest='group')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')

    args, _ = parser.parse_known_args()

    train_set = PointDataSet(args.pos, args.neg, args.group, True, 5)
    test_set = PointDataSet(args.pos, args.neg, args.group, False, 5)
    # train_loader, test_loader = load_data(args.pos, args.neg, args.n_lmk, args.k, args.load, args.cv)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = Net(args.keypoint, args.network).to(device)
    model = pointnet2_cls_ssg.get_model(2, normal_channel=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = pointnet2_cls_ssg.get_loss()

    res_test_acc = []

    for epoch in range(1, args.n_epoch+1):
        train_loss = train(train_loader)
        train_acc, val_loss = train_(val_loader)
        test_acc, out, test_loss = test(test_loader)

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