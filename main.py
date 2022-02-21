'''
Rechoose pseudo labels every stage
'''
import argparse
import numpy as np
import torch
import torch.optim as optim
import random
from utils import accuracy
from utils import *
from utils_plot import *
import torch.nn as nn
import os


global result
result = []
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--dataset', type=str, default="Cora",
                    help='dataset for training')
parser.add_argument('--labelrate', type=int, required=True, default=20)
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--stage', type=int, default=3)
parser.add_argument('--threshold', type=float, default=0.53)
parser.add_argument('--beta', type=float, default=1/3,
                    help='coefficient for weighted CE loss')
parser.add_argument('--drop_method', type=str, required=True, default='dropout')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--droprate', type=float, default=0.5,
                    help='Droprate for MC-Dropout')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
criterion = torch.nn.CrossEntropyLoss().cuda()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

def train(model_path, idx_train, idx_val, idx_test, features, adj, pseudo_labels, labels, bald, T, g, seed):
    sign = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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
        output = torch.softmax(output, dim=1)
        output = torch.mm(output, T)
        sign = False
        loss_train = weighted_cross_entropy(output[idx_train], pseudo_labels[idx_train], bald[idx_train], args.beta, nclass, sign)
        # loss_train = criterion(output[idx_train], pseudo_labels[idx_train])
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
            torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False)
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
    g, adj, features, labels, idx_train, idx_val, idx_test, oadj = load_data(dataset, args.labelrate)
    g = g.to(device)
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    train_index = torch.where(idx_train)[0]
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    idx_pseudo = torch.zeros_like(idx_train)
    n_node = labels.size()[0]
    nclass = labels.max().item() + 1

    if args.drop_method == 'dropedge':
        mc_adj = get_mc_adj(oadj, device, args.droprate)

    if args.labelrate != 20:
        idx_train[train_index] = True
        idx_train = generate_trainmask(idx_train, idx_val, idx_test, n_node, nclass, labels, args.labelrate)

    idx_train_ag = idx_train.clone().to(device)
    pseudo_labels = labels.clone().to(device)
    bald = torch.ones(n_node).to(device)
    T = nn.Parameter(torch.eye(nclass, nclass).to(device)) # transition matrix
    T.requires_grad = False
    seed = np.random.randint(0, 10000)
    for s in range(args.stage):
        best_output = train(model_path, idx_train_ag, idx_val, idx_test, features, adj, pseudo_labels, labels, bald, T, g, seed)
        T = update_T(best_output, idx_train, labels, T, device)
        idx_unlabeled = ~(idx_train | idx_test | idx_val)
        if args.drop_method == 'dropout':
            bald = uncertainty_dropout(adj, features, nclass, model_path, args, device)
        elif args.drop_method == 'dropedge':
            bald = uncertainty_dropedge(mc_adj, adj, features, nclass, model_path, args, device)

        # generate pseudo labels
        state_dict = torch.load(model_path)
        model = get_models(args, features.shape[1], nclass, g=g)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        best_output = model(features, adj)
        idx_train_ag, pseudo_labels, idx_pseudo = regenerate_pseudo_label(best_output, labels, idx_train, idx_unlabeled,
                                                                          args.threshold, device)
        # Testing
        acc_test, loss_test = test(adj, features, labels, idx_test, nclass, model_path, g)

        # plot_data_distribution(best_output.detach().cpu(), bald, labels, idx_unlabeled, idx_train, idx_test, args.dataset)
        # plot_un_conf(best_output.detach().cpu(), labels, idx_unlabeled, bald, args.dataset)
        # plot_conf_dis(best_output, idx_unlabeled, labels, args.dataset, s)
        # plot_dis_pseudo(dataset, best_output.detach(), idx_train, idx_test, labels, idx_unlabeled, idx_pseudo, bald, s)


    return



if __name__ == '__main__':
    model_path = './save_model/%s-%s-%d-%f-%f-%f-%s.pth' % (
                    args.model, args.dataset, args.labelrate, args.threshold, args.beta, args.droprate, args.drop_method)
    main(args.dataset, model_path)


