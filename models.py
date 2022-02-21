import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from dgl.nn.pytorch.conv import SAGEConv, APPNPConv, GINConv, SGConv, GATConv
import dgl.function as fn
from layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.gc2(x, adj)
        return x
    def get_emb(self, x, adj):
        return self.gc1(x, adj)

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs, adj):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits

class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None)) # activation None

    def forward(self, features, adj):
        h = features
        for layer in self.layers:
            h = layer(self.g, h)
        return F.log_softmax(h)


class APPNP(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 hiddens,
                 n_classes,
                 activation,
                 dropout,
                 alpha,
                 k):
        super(APPNP, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats, hiddens))
        self.layers.append(nn.Linear(hiddens, n_classes))
        self.activation = activation
        self.feat_drop = nn.Dropout(dropout)
        self.propagate = APPNPConv(k, alpha, dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, features, adj):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(self.g, h)
        return h

class GIN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 hidden,
                 n_classes,
                 activation,
                 feat_drop,
                 eps):
        super(GIN, self).__init__()
        self.g = g

        self.mlp1 = nn.Linear(in_feats, hidden)
        self.mlp2 = nn.Linear(hidden, n_classes)

        self.layer1 = GINConv(self.mlp1, 'sum', eps)
        self.layer2 = GINConv(self.mlp2, 'sum', eps)

        self.activation = activation

        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

    def forward(self, features, adj):
        # prediction step
        h = features
        h = self.feat_drop(h)

        h = self.layer1(self.g, h)
        h = self.activation(h)
        h = self.feat_drop(h)

        h = self.layer2(self.g, h)
        return h

class SGC(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_classes,
                 num_k):
        super(SGC, self).__init__()
        self.g = g

        self.model = SGConv(in_feats,
                   n_classes,
                   k=num_k,
                   cached=True)


    def forward(self, features, adj):
        # prediction step
        h = self.model(self.g, features)

        return h


class MixHopConv(nn.Module):
    r"""
    Description
    -----------
    MixHop Graph Convolutional layer from paper `MixHop: Higher-Order Graph Convolutional Architecturesvia Sparsified Neighborhood Mixing
     <https://arxiv.org/pdf/1905.00067.pdf>`__.
    .. math::
        H^{(i+1)} =\underset{j \in P}{\Bigg\Vert} \sigma\left(\widehat{A}^j H^{(i)} W_j^{(i)}\right),
    where :math:`\widehat{A}` denotes the symmetrically normalized adjacencymatrix with self-connections,
    :math:`D_{ii} = \sum_{j=0} \widehat{A}_{ij}` its diagonal degree matrix,
    :math:`W_j^{(i)}` denotes the trainable weight matrix of different MixHop layers.
    Parameters
    ----------
    in_dim : int
        Input feature size. i.e, the number of dimensions of :math:`H^{(i)}`.
    out_dim : int
        Output feature size for each power.
    p: list
        List of powers of adjacency matrix. Defaults: ``[0, 1, 2]``.
    dropout: float, optional
        Dropout rate on node features. Defaults: ``0``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    batchnorm: bool, optional
        If True, use batch normalization. Defaults: ``False``.
    """

    def __init__(self,
                 in_dim,
                 out_dim,
                 p=[0, 1, 2],
                 dropout=0,
                 activation=None,
                 batchnorm=False):
        super(MixHopConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p = p
        self.activation = activation
        self.batchnorm = batchnorm

        # define dropout layer
        self.dropout = nn.Dropout(dropout)

        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim * len(p))

        # define weight dict for each power j
        self.weights = nn.ModuleDict({
            str(j): nn.Linear(in_dim, out_dim, bias=False) for j in p
        })

    def forward(self, graph, feats):
        with graph.local_scope():
            # assume that the graphs are undirected and graph.in_degrees() is the same as graph.out_degrees()
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).to(feats.device).unsqueeze(1)
            max_j = max(self.p) + 1
            outputs = []
            for j in range(max_j):

                if j in self.p:
                    output = self.weights[str(j)](feats)
                    outputs.append(output)

                feats = feats * norm
                graph.ndata['h'] = feats
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                feats = graph.ndata.pop('h')
                feats = feats * norm

            final = torch.cat(outputs, dim=1)

            if self.batchnorm:
                final = self.bn(final)

            if self.activation is not None:
                final = self.activation(final)

            final = self.dropout(final)

            return final

class MixHop(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers=2,
                 p=[0, 1, 2],
                 input_dropout=0.0,
                 layer_dropout=0.0,
                 activation=None,
                 batchnorm=False):
        super(MixHop, self).__init__()
        self.g=g
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.p = p
        self.input_dropout = input_dropout
        self.layer_dropout = layer_dropout
        self.activation = activation
        self.batchnorm = batchnorm

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(self.input_dropout)

        # Input layer
        self.layers.append(MixHopConv(self.in_dim,
                                      self.hid_dim,
                                      p=self.p,
                                      dropout=self.input_dropout,
                                      activation=self.activation,
                                      batchnorm=self.batchnorm))

        # Hidden layers with n - 1 MixHopConv layers
        for i in range(self.num_layers - 2):
            self.layers.append(MixHopConv(self.hid_dim * len(self.p),
                                          self.hid_dim,
                                          p=self.p,
                                          dropout=self.layer_dropout,
                                          activation=self.activation,
                                          batchnorm=self.batchnorm))

        self.fc_layers = nn.Linear(self.hid_dim * len(self.p), self.out_dim, bias=False)

    def forward(self, feats, adj):
        feats = self.dropout(feats)
        for layer in self.layers:
            feats = layer(self.g, feats)

        feats = self.fc_layers(feats)

        return feats


class MLP(nn.Module):
    def __init__(self, nfeat, nclass, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(nfeat, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, nclass)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.fc3(output)
        return output


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=20):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece +=  torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, n_bins):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))
        self.n_bins = n_bins

    def forward(self, features, adj):
        logits = self.model(features, adj)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_parameters(self, features, adj, labels, idx_val, lr=0.01, max_iter=200):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss(self.n_bins).cuda()

        # First: collect all the logits and labels for the validation set

        with torch.no_grad():
            logits = self.model(features, adj)

        logits = logits[idx_val]
        labels = labels[idx_val]

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self

