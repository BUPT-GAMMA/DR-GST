import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import math
from sklearn.isotonic import IsotonicRegression
import seaborn as sns
import networkx as nx
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import matplotlib as mpl
from sklearn.datasets import make_circles
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from utils import get_confidence


def plot_conf_dis(output, mask, labels, dataset, stage, sign=True):
    if sign:
        pred = torch.softmax(output[mask], dim=1).cpu()
    else:
        pred = output[mask]
    confidence, pred_max_index = torch.max(pred, 1)
    correct_index = (labels[mask].cpu() == pred_max_index).cpu()
    plot_histograms(confidence[correct_index], confidence[np.invert(correct_index)],
                    'Conf. - %s - %d'%(dataset, stage), ['Correct', 'Incorrect'], ['Confidence', 'Density'], 50)

def plot_un_conf(output, labels, mask, bald, dataset):
    bald = bald[mask]
    output = output[mask]
    labels = labels[mask]
    confidence, pred = get_confidence(output)
    sign = (pred == labels).cpu()
    # bald = bald / torch.sum(bald)
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams["font.weight"] = "bold"
    plt.scatter(confidence[sign].cpu(), bald[sign].cpu(), label='Correct')
    plt.scatter(confidence[~sign].cpu(), bald[~sign].cpu(), label='Incorrect')
    plt.legend(fontsize=14)
    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('Information Gain', fontsize=14)
    plt.title(dataset, fontsize=16, fontweight="bold")
    plt.tick_params(labelsize=13)
    plt.plot()
    plt.savefig('output/un_conf_' + dataset + '.png', format='png', transparent=True, dpi=300,
                pad_inches=0, bbox_inches='tight')
    plt.show()

def plot_data_distribution(output, bald, labels, idx_unlabeled, idx_train, idx_test, dataset):
    data = TSNE().fit_transform(output[idx_train | idx_unlabeled].cpu())
    confidence = get_confidence(output)[0]
    plt.scatter(data[:, 0], data[:, 1], c=bald[idx_train | idx_unlabeled].cpu())
    # plt.scatter(data[:,0], data[:,1], c=confidence[idx_train | idx_unlabeled].cpu())
    plt.colorbar()
    plt.title(dataset, fontsize=16, fontweight="bold")
    plt.savefig('output/TSNE_'+dataset+'.png', format='png', transparent=True, dpi=300,
                pad_inches=0, bbox_inches='tight')
    plt.show()

def Gauss_digram():
    x = np.random.normal(0, 0.1, 2000)
    y = np.random.normal(0, 0.1, 2000)
    x_loose = np.random.normal(0, 0.3, 500)
    y_loose = np.random.normal(0, 0.3, 500)
    # x_out, y_out = make_circles(n_samples=2000, factor=0.99, noise=0.2)
    # g = sns.jointplot(np.hstack((x,x_loose)), np.hstack((y,y_loose)), xlim=(-1,1), ylim=(-1,1))
    # sns.scatterplot(x_out[:, 0], x_out[:, 1], color='#738595', ax=g.ax_joint)
    # plt.tick_params(labelsize=13)
    # plt.rcParams['axes.labelweight'] = 'bold'
    # plt.rcParams["font.weight"] = "bold"
    # plt.plot()
    # plt.savefig('output/gauss_2.png', format='png', transparent=True, dpi=300,
    #             pad_inches=0, bbox_inches='tight')
    # plt.show()

    # kde plot
    x_out, y_out = make_circles(n_samples=900, factor=0.1, noise=0.8)
    x_out_1, y_out_1 = make_circles(n_samples=3000, factor=0.9, noise=0.25)
    sns.scatterplot(x_out_1[:, 0], x_out_1[:, 1], color='#738595')
    plt.show()
    sns.kdeplot(np.hstack((x_out[:,0],x_out_1[:,0])),np.hstack((x_out[:,1],x_out_1[:,1])),shade=True)
    plt.xlim(-0.9,0.9)
    plt.ylim(-0.9,0.9)
    plt.savefig('output/ratio.png', format='png', transparent=True, dpi=300,
                             pad_inches=0, bbox_inches='tight')
    plt.show()
    return


def plot_dis_pseudo(dataset, output, idx_train, idx_test, labels, idx_unlabeled, idx_pseudo, bald, stage):
    data = TSNE().fit_transform(output[idx_unlabeled].cpu())
    bald[~(idx_pseudo)] = 0
    plt.scatter(data[:, 0], data[:, 1], c=bald[idx_unlabeled].cpu())
    plt.colorbar()
    # plt.title(dataset, fontsize=16, fontweight="bold")
    plt.savefig('output/TSNE_pseudo_'+str(stage)+'_' + dataset + '.png', format='png', transparent=True, dpi=300,
                pad_inches=0, bbox_inches='tight')
    plt.show()
