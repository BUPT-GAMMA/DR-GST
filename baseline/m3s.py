'''
Once remarked, the unlabeled data will not be remarked in next stages
'''
import torch
import os
import sys
import argparse
import numpy as np
import torch.optim as optim
from utils_baseline import *

os_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os_path)

from utils import accuracy
from utils import *


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--dataset', type=str, default="Pubmed",
                    help='dataset for training')
parser.add_argument('--n_cluster', type=int, default=200,
                    help='Number of clusters')
parser.add_argument('--max_iter', type=int, default=20,
                    help='Number of epochs for kmeans')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--stage', type=int, default=10)
parser.add_argument('--threshold', type=float, default=0.7)
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

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
criterion = torch.nn.CrossEntropyLoss().cuda()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

def train(model_path, idx_train, idx_val, idx_test, features, adj, pseudo_labels, labels, seed):
    sign = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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
        loss_train = criterion(output[idx_train], pseudo_labels[idx_train])
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
    seed = np.random.randint(0, 10000)
    for s in range(args.stage):
        best_output = train(model_path, idx_train_ag, idx_val, idx_test, features, adj, pseudo_labels, labels, seed)
        idx_unlabeled = ~(idx_train_ag | idx_test | idx_val)
        state_dict = torch.load(model_path)
        model = get_models(args, features.shape[1], nclass)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        best_output = model(features, adj)
        # Centroids of class in labeled data
        centroid_train = Centroid_train(best_output[idx_train_ag], labels[idx_train_ag], nclass, device)
        # Kmeans
        kmeans = Kmeans(args.n_cluster, args.max_iter, False, device)
        kmeans.fit(best_output[idx_unlabeled])
        # Generating the pseudo labels for kmeans
        kmeans_pseudo_labels = kmeans.aligning_labels(centroid_train)
        # Generating the pseudo labels
        idx_train_ag, pseudo_labels = generate_pseudo_label_after_aligning(best_output, labels, idx_train_ag, idx_unlabeled,
                                                                          kmeans_pseudo_labels, args.threshold, device)
        # Testing
        acc_test, loss_test = test(adj, features, labels, idx_test, nclass, model_path)

    return



if __name__ == '__main__':
    model_path = os_path + '/save_model/baseline/m3s-%s-%s-%d-%f.pth' % (args.model, args.dataset, args.labelrate, args.threshold)
    main(args.dataset, model_path)