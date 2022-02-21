import os
import sys
import argparse
import numpy as np
import torch
import torch.optim as optim

os_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os_path)

from utils import accuracy
from utils import *
from utils_plot import *
import torch.nn as nn
import os

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='MixHop')
parser.add_argument('--dataset', type=str, default="Cora",
                    help='dataset for training')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--n_run', type=int, default=10,
                    help='Number of repeated experiments')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--labelrate', type=int, default=20)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--nb_out_heads', type=int, default=1,
                    help='Number of output head attentions')
parser.add_argument('--residual', type=bool, default=True)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
criterion = torch.nn.CrossEntropyLoss().cuda()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

def train(model_path, idx_train, idx_val, idx_test, features, adj, labels, g):
    nclass = labels.max().item() + 1
    # Model and optimizer
    model = get_models(args, features.shape[1], nclass, g=g)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)
    best, bad_counter = 100, 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = criterion(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
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
def test(adj, features, labels, idx_test, nclass, model_path, g):
    nfeat = features.shape[1]
    state_dict = torch.load(model_path)
    model = get_models(args, features.shape[1], nclass, g=g)
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
    g = g.to(device)
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

    best_output = train(model_path, idx_train, idx_val, idx_test, features, adj, labels, g)
    # Testing
    acc_test, loss_test = test(adj, features, labels, idx_test, nclass, model_path, g)

    return


if __name__ == '__main__':
    model_path = os_path + '/save_model/baseline/gnns-%s-%s-%d.pth' % (args.model, args.dataset, args.labelrate)
    main(args.dataset, model_path)


