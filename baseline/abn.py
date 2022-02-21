import os
import sys
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from utils_baseline import *

os_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os_path)

from utils import accuracy
from utils import *
from utils_plot import *
import torch.nn as nn

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--dataset', type=str, default="Cora",
                    help='dataset for training')
parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--Lambda', type=float, default=1)
parser.add_argument('--Lambda1', type=float, default=0)
parser.add_argument('--n_positive', type=int, default=2)
parser.add_argument('--n_negative_per_positive', type=int, default=5)
parser.add_argument('--balance', type=bool, default=False)
parser.add_argument('--threshold', type=float, default=0.65)
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.8,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--labelrate', type=int, default=20)
parser.add_argument('--patience', type=int, default=100)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
criterion = torch.nn.CrossEntropyLoss().cuda()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

def train(model_path, idx_train, idx_val, idx_test, features, adj, oadj, pseudo_labels, labels):
    sign = True
    nclass = labels.max().item() + 1
    # Model and optimizer
    model = get_models(args, features.shape[1], nclass)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)
    best, bad_counter = 100, 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        # Generating pseudo labels
        idx_unlabeled = ~(idx_train | idx_test | idx_val)
        idx_train_ag, pseudo_labels, idx_pseudo = regenerate_pseudo_label(output, labels, idx_train, idx_unlabeled,
                                                                          args.threshold, device)

        # Neg Loss
        positive_index, negative_index = negative_sampling(idx_train_ag, args, oadj)
        n_out = output[negative_index]
        n_out = torch.softmax(n_out, dim=1)
        n_out = n_out - (1e-4)
        n_out = 1 - n_out
        n_out = torch.log(n_out)
        y = pseudo_labels[positive_index]
        y = y.repeat_interleave(args.n_negative_per_positive, 0)
        L_nsr = F.nll_loss(n_out, y)

        # Loss of unlabeled data
        if idx_pseudo.sum() == 0:
            L_un = 0
        elif args.balance:
            pseudo_index = torch.where(idx_pseudo)[0]
            n_per_class = []
            for i in range(nclass):
                tmp = (pseudo_labels[pseudo_index] == i).sum()
                n_per_class.append(tmp.item())
            n_per_class = torch.tensor([1/i if i != 0 else 0 for i in n_per_class])
            balance_cof = n_per_class[pseudo_labels[idx_pseudo]].to(device)
            tmp_pseuo_labels = F.one_hot(pseudo_labels[idx_pseudo], nclass)
            tmp_output = F.log_softmax(output[idx_pseudo], dim=1)
            L_un = - tmp_output * tmp_pseuo_labels * balance_cof.unsqueeze(dim=1)
            L_un = torch.mean(L_un) * nclass

        else:
            L_un = criterion(output[idx_pseudo], pseudo_labels[idx_pseudo])

        loss_train = criterion(output[idx_train], pseudo_labels[idx_train]) + \
                     L_un * args.Lambda + L_nsr * args.Lambda1 / (args.n_positive * args.n_negative_per_positive)

        acc_train = accuracy(output[idx_train], pseudo_labels[idx_train])
        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            output = model(features, adj)
            loss_val = criterion(output[idx_val], labels[idx_val])
            loss_test = criterion(output[idx_test], labels[idx_test])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            acc_test = accuracy(output[idx_test], labels[idx_test])

        if loss_val < best:
            torch.save(model.state_dict(), model_path)
            best = loss_val
            bad_counter = 0
            best_output = output
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

        # print(f'epoch: {epoch}',
        #       f'loss_train: {loss_train.item():.4f}',
        #       f'acc_train: {acc_train:.4f}',
        #       f'loss_val: {loss_val.item():.4f}',
        #       f'acc_val: {acc_val:.4f}',
        #       f'loss_test: {loss_test.item():4f}',
        #       f'acc_test: {acc_test:.4f}')
    return best_output


@torch.no_grad()
def test(adj, features, labels, idx_test, nclass, model_path):
    nfeat = features.shape[1]
    state_dict = torch.load(model_path)
    model = get_models(args, features.shape[1], nclass)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    output = model(features, adj)
    loss_test = criterion(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print(f"Test set results",
          f"loss= {loss_test.item():.4f}",
          f"accuracy= {acc_test:.4f}")

    return acc_test, loss_test


def main(dataset, model_path):
    g, adj, features, labels, idx_train, idx_val, idx_test, oadj = load_data(dataset, args.labelrate, os_path)
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    train_index = torch.where(idx_train)[0]
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    n_node = labels.size()[0]
    nclass = labels.max().item() + 1


    if args.labelrate != 20:
        idx_train[train_index] = True
        idx_train = generate_trainmask(idx_train, idx_val, idx_test, n_node, nclass, labels, args.labelrate)

    idx_train_ag = idx_train.clone().to(device)
    pseudo_labels = labels.clone().to(device)
    best_output = train(model_path, idx_train_ag, idx_val, idx_test, features, adj, oadj, pseudo_labels, labels)
    # Testing
    acc_test, loss_test = test(adj, features, labels, idx_test, nclass, model_path)

    return



if __name__ == '__main__':
    model_path = os_path + '/save_model/baseline/abn-%s-%s-%d-%f.pth' % (args.model, args.dataset, args.labelrate, args.threshold)
    main(args.dataset, model_path)


