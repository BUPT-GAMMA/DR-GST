import numpy as np
import scipy
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import networkx as nx
import torch.optim as optim
import dgl
from dgl.data import CiteseerGraphDataset
from dgl.data import CoraGraphDataset
from dgl.data import PubmedGraphDataset
from dgl.data import CoraFullDataset
from dgl.data import CoauthorCSDataset
from dgl.data import  CoauthorPhysicsDataset
from dgl.data import  AmazonCoBuyComputerDataset
import random
from models import *
from sklearn.metrics.pairwise import cosine_similarity as cos




def load_data(dataset ,labelrate, os_path=None):
    citation_data = ['Cora', 'Citeseer', 'Pubmed']
    if dataset == 'Cora':
        data = CoraGraphDataset()
    elif dataset == 'Citeseer':
        data = CiteseerGraphDataset()
    elif dataset == 'Pubmed':
        data = PubmedGraphDataset()
    elif dataset == 'CoraFull':
        data = CoraFullDataset()
    elif dataset == 'CaCS':
        data = CoauthorCSDataset()
    elif dataset == 'CaPH':
        data = CoauthorPhysicsDataset()
    elif dataset == 'ACom':
        data = AmazonCoBuyComputerDataset()
    else:
        # 因为我们在后面重新生成train_mask，这里直接load 20即可
        return load_local_data(dataset, 20, os_path)

    g = data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']

    if dataset in citation_data:
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
    else:
        train_mask, val_mask, test_mask = generate_mask(dataset, labels, 20, os_path)
    nxg = g.to_networkx()
    adj = nx.to_scipy_sparse_matrix(nxg, dtype=np.float)
    oadj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = preprocess_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return g, adj, features, labels, train_mask, val_mask, test_mask, oadj

def generate_mask(dataset, labels, labelrate, os_path=None):
    datalength = labels.size()
    train_mask, val_mask, test_mask = torch.full((1, datalength[0]), fill_value=False, dtype=bool), torch.full(
        (1, datalength[0]), fill_value=False, dtype=bool) \
        , torch.full((1, datalength[0]), fill_value=False, dtype=bool)
    path = 'data/%s/' % (dataset)
    if os_path != None:
        path = os_path + '/' + path
    mask = (train_mask, val_mask, test_mask)
    name = ('train', 'val', 'test')
    for (i, na) in enumerate(name):
        with open(path + na + '%s.txt' % labelrate, 'r') as f:
            index = f.read().splitlines()
            index = list(map(int, index))
            mask[i][0][index] = 1
    return mask[0][0], mask[1][0], mask[2][0]

def generate_trainmask(train_mask, val_mask, test_mask, n_node, nclass, labels, labelrate):
    train_index = torch.where(train_mask)[0]
    train_mask = train_mask.clone()
    train_mask[:] = False
    label = labels[train_index]
    for i in range(nclass):
        class_index = torch.where(label == i)[0].tolist()
        class_index = random.sample(class_index, labelrate)
        train_mask[train_index[class_index]] = True
    return train_mask

def load_local_data(dataset, labelrate, os_path=None):
    feature_path = 'data/%s/%s.feature' % (dataset, dataset)
    label_path = 'data/%s/%s.label' % (dataset, dataset)
    edge_path = 'data/%s/%s.edge' % (dataset, dataset)
    if os_path != None:
        feature_path, label_path, edge_path = os_path+'/'+feature_path, os_path+'/'+label_path, os_path+'/'+edge_path
    f = np.loadtxt(feature_path, dtype=float)
    l = np.loadtxt(label_path, dtype=int)
    labels = torch.LongTensor(np.array(l))
    datalength = labels.size()
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))

    train_mask, val_mask, test_mask = generate_mask(dataset, labels, labelrate, os_path)

    struct_edges = np.genfromtxt(edge_path, dtype=np.int32)
    edges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(datalength[0], datalength[0]),
                         dtype=np.float32)
    g = dgl.from_scipy(adj)
    oadj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = preprocess_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return g, adj, features, labels, train_mask, val_mask, test_mask, oadj


def preprocess_adj(adj, with_ego=True):
    """Preprocessing of adjacency matrix for simple GCN model and conversion
    to tuple representation."""
    if with_ego:
        adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    else:
        adj_normalized = normalize_adj(adj)
    return adj_normalized


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # D^-0.5AD^0.5

def load_model(model_path, args, nfeature, nclass, device):
    state_dict = torch.load(model_path)
    model = get_models(args, nfeature, nclass)
    model.load_state_dict(state_dict)
    model.to(device)
    return model

def accuracy(pred, targ):
    pred = torch.softmax(pred, dim=1)
    pred_max_index = torch.max(pred, 1)[1]
    ac = ((pred_max_index == targ).float()).sum().item() / targ.size()[0]
    return ac


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_models(args, nfeat, nclass, sign=True, g=None):
    model_name = args.model
    if sign:
        droprate = args.dropout
    else:
        droprate = args.droprate
    if model_name == 'GCN':
        model = GCN(nfeat=nfeat,
                    nhid=args.hidden,
                    nclass=nclass,
                    dropout=droprate)
    elif model_name == 'GAT':
        model = GAT(g=g,
                    num_layers=1,
                    in_dim=nfeat,
                    num_hidden=args.hidden,
                    num_classes=nclass,
                    heads=([args.nb_heads] * 1) + [args.nb_out_heads],
                    activation=F.relu,
                    feat_drop=0.6,
                    attn_drop=0.6,
                    negative_slope=args.alpha,
                    residual=args.residual)
    elif model_name == 'GraphSAGE':
        model = GraphSAGE(g=g,
                          in_feats=nfeat,
                          n_hidden=args.hidden,
                          n_classes=nclass,
                          activation=F.relu,
                          dropout=droprate,
                          aggregator_type='gcn')
    elif model_name == 'APPNP':
        model = APPNP(g=g,
                      in_feats=nfeat,
                      hiddens=args.hidden,
                      n_classes=nclass,
                      activation=F.relu,
                      dropout=droprate,
                      alpha=0.1,
                      k=10)
    elif model_name == 'GIN':
        model = GIN(g=g,
                    in_feats=nfeat,
                    hidden=args.hidden,
                    n_classes=nclass,
                    activation=F.relu,
                    feat_drop=droprate,
                    eps=0.2)
    elif model_name == 'SGC':
        model = SGC(g=g,
                    in_feats=nfeat,
                    n_classes=nclass,
                    num_k=2)
    elif model_name == 'MixHop':
        model = MixHop(g=g,
                       in_dim=nfeat,
                       hid_dim=args.hidden,
                       out_dim=nclass,
                       num_layers=args.num_layers,
                       input_dropout=args.dropout,
                       layer_dropout=0.9,
                       activation=torch.tanh,
                       batchnorm=True)

    return model

def get_confidence(output, with_softmax=False):

    if not with_softmax:
        output = torch.softmax(output, dim=1)

    confidence, pred_label = torch.max(output, dim=1)

    return confidence, pred_label

def generate_pseudo_label(output, pseudo_labels, idx_train, idx_val, idx_test, threshold, sign=False):
    train_index = torch.where(idx_train==True)
    test_index = torch.where(idx_test==True)
    val_index = torch.where(idx_val==True)
    confidence, pred_label = get_confidence(output, sign)
    index = torch.where(confidence>threshold)[0]
    for i in index:
        if i not in train_index[0] and i not in test_index[0] and i not in val_index[0]:
            pseudo_labels[i] = pred_label[i]
            idx_train[i] = True
    return idx_train, pseudo_labels

def regenerate_pseudo_label(output, labels, idx_train, unlabeled_index, threshold, device, sign=False):
    're-generate pseudo labels every stage'
    unlabeled_index = torch.where(unlabeled_index == True)[0]
    confidence, pred_label = get_confidence(output, sign)
    index = torch.where(confidence > threshold)[0]
    pseudo_index = []
    pseudo_labels, idx_train_ag = labels.clone().to(device), idx_train.clone().to(device)
    for i in index:
        if i not in idx_train:
            pseudo_labels[i] = pred_label[i]
            # pseudo_labels[i] = labels[i]
            if i in unlabeled_index:
                idx_train_ag[i] = True
                pseudo_index.append(i)
    idx_pseudo = torch.zeros_like(idx_train)
    pseudo_index = torch.tensor(pseudo_index)
    if pseudo_index.size()[0] != 0:
        idx_pseudo[pseudo_index] = 1
    return idx_train_ag, pseudo_labels, idx_pseudo


def construct_graph(similarity, topk, length):
    inds, inds_inverse = [[], []], [[], []]
    for i in range(similarity.shape[0]):
        if (similarity[i] == 0).sum() != 0:
            index = np.where(similarity[i] == 0)[0]
            ind = np.random.choice(index, size=topk).tolist()
        else:
            ind = np.argpartition(similarity[i], topk)[:topk].tolist()
        inds[0] += [i] * topk
        inds_inverse[1] += [i] * topk
        inds[1] += ind
        inds_inverse[0] += ind
    fadj = torch.zeros(length, length)
    fadj[inds] = 1
    fadj[inds_inverse] = 1
    return fadj

def generate_fusion_knn(features, emb1, topk, length, nsample):
    sample_index = random.sample(range(nsample), length)
    sample_index.sort()
    tmp_adj1 = construct_graph(cos(features[sample_index].cpu()), topk, length)
    tmp_adj2 = construct_graph(cos(emb1[sample_index].detach().cpu().numpy()), topk, length)
    fadj = (tmp_adj1.int() | tmp_adj2.int()).float()
    fadj = preprocess_adj(fadj, False)
    fadj = sparse_mx_to_torch_sparse_tensor(fadj)
    return sample_index, fadj


def uncertainty_dropout(adj, features, nclass, model_path, args, device):
    f_pass = 100
    state_dict = torch.load(model_path)
    model = get_models(args, features.shape[1], nclass, False)
    model.load_state_dict(state_dict)
    model.to(device)
    out_list = []
    with torch.no_grad():
        for _ in range(f_pass):
            output = model(features, adj)
            output = torch.softmax(output, dim=1)
            out_list.append(output)
        out_list = torch.stack(out_list)
        out_mean = torch.mean(out_list, dim=0)
        entropy = torch.sum(torch.mean(out_list * torch.log(out_list), dim=0), dim=1)
        Eentropy = torch.sum(out_mean * torch.log(out_mean), dim=1)
        bald = entropy - Eentropy
    return bald

def uncertainty_dropout_feature(adj, features, nclass, model_path, args, device):
    f_pass = 100
    state_dict = torch.load(model_path)
    model = get_models(args, features.shape[1], nclass, False)
    model.load_state_dict(state_dict)
    model.to(device)
    out_list = []
    with torch.no_grad():
        model.eval()
        for _ in range(f_pass):
            features_tmp = features.clone()
            features_tmp = F.dropout(features_tmp, p = args.droprate)
            output = model(features_tmp, adj)
            output = torch.softmax(output, dim=1)
            out_list.append(output)
        out_list = torch.stack(out_list)
        out_mean = torch.mean(out_list, dim=0)
        entropy = torch.sum(torch.mean(out_list * torch.log(out_list), dim=0), dim=1)
        Eentropy = torch.sum(out_mean * torch.log(out_mean), dim=1)
        bald = entropy - Eentropy
    return bald

def get_mc_adj(oadj, device, droprate=0.1):
    f_pass = 100
    edge_index = oadj.coalesce().indices()
    mc_adj = []
    for i in range(f_pass):
        adj_tmp = oadj.clone().to_dense()
        drop = np.random.random(edge_index.size()[1])
        drop = np.where(drop < droprate)[0]
        edge_index_tmp = edge_index[:, drop]
        adj_tmp[edge_index_tmp[0], edge_index_tmp[1]] = 0
        adj_tmp = preprocess_adj(sp.coo_matrix(adj_tmp))
        adj_tmp = sparse_mx_to_torch_sparse_tensor(adj_tmp).to(device)
        mc_adj.append(adj_tmp)
    return mc_adj

def uncertainty_dropedge(mc_adj, adj, features, nclass, model_path, args, device):
    state_dict = torch.load(model_path)
    model = get_models(args, features.shape[1], nclass)
    model.load_state_dict(state_dict)
    model.to(device)
    out_list = []
    with torch.no_grad():
        model.eval()
        for madj in mc_adj:
            output = model.gc1(features, adj)
            output = torch.relu(output)
            output = model.gc2(output, madj)
            output = torch.softmax(output, dim=1)
            output = output + 1e-15
            out_list.append(output)
        out_list = torch.stack(out_list)
        out_mean = torch.mean(out_list, dim=0)
        entropy = torch.sum(torch.mean(out_list * torch.log(out_list), dim=0), dim=1)
        Eentropy = torch.sum(out_mean * torch.log(out_mean), dim=1)
        bald = entropy - Eentropy
    return bald


def weighted_cross_entropy(output, labels, bald, beta, nclass, sign=True):
    bald += 1e-6
    if sign:
        output = torch.softmax(output, dim=1)
    bald = bald / (torch.mean(bald) * beta)
    labels = F.one_hot(labels, nclass)
    loss = -torch.log(torch.sum(output * labels, dim=1))
    loss = torch.sum(loss * bald)
    loss /= labels.size()[0]
    return loss
def update_T(output, idx_train, labels, T, device):
    output = torch.softmax(output, dim=1)
    T.requires_grad = True
    optimizer = optim.Adam([T], lr=0.01, weight_decay=5e-4)
    mse_criterion = torch.nn.MSELoss().cuda()
    index = torch.where(idx_train)[0]
    nclass = labels.max().item() + 1
    for epoch in range(200):
        optimizer.zero_grad()
        loss = mse_criterion(output[index], T[labels[index]]) + mse_criterion(T, torch.eye(nclass).to(device))
        loss.backward()
        optimizer.step()
    T.requires_grad = False
    return T
