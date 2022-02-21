import torch
import  random

class Kmeans:
    def __init__(self, n_clusters=20, max_iter=None, verbose=True,device = torch.device("cpu")):

        self.n_cluster = n_clusters
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit(self, x):
        init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
        init_points = x[init_row]
        self.centers = init_points
        while True:
            self.nearest_center(x) # To label the samples
            self.update_center(x) # To Update the centroid of cluster
            if self.count == self.max_iter:
                break
            self.count += 1

        self.representative_sample()

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        for i, sample in enumerate(x):
            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), (1))
            labels[i] = torch.argmin(dist)
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
        self.labels = labels
        self.dists = dists

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device)
        for i in range(self.n_clusters):
            mask = self.labels == i
            if mask.sum() == 0:
                continue
            cluster_samples = x[mask]
            centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))
        self.centers = centers
        self.n_clusters = self.centers.size()[0]

    def aligning_labels(self, centroid_train):
        x_size, y_size = self.centers.size()
        centers = self.centers.reshape([x_size,1,y_size])
        centers = centers.expand([x_size, y_size,y_size])
        dists = torch.mean((centers - centroid_train)**2,dim=-1)
        pseudo_centroid_labels = torch.argmin(dists, dim=1)
        pseudo_labels = pseudo_centroid_labels[self.labels]
        return pseudo_labels

    def representative_sample(self):
        # To find the samples nearest to the centroid of cluster
        self.representative_samples = torch.argmin(self.dists, (0))

def generate_pseudo_label_after_aligning(output, pseudo_labels, idx_train_ag, unlabeled_index, kmeans_tmp_labels, threshold, device):
    unlabeled_index = torch.where(unlabeled_index == True)[0]
    kmeans_labels = torch.zeros_like(pseudo_labels)
    kmeans_labels[unlabeled_index] = kmeans_tmp_labels
    output = torch.softmax(output, dim=1)
    confidence, pred_label = torch.max(output, dim=1)
    indices = torch.where(confidence>threshold)[0]
    for index in indices:
        if index in unlabeled_index and pred_label[index] == kmeans_labels[index]:
            pseudo_labels[index] = pred_label[index]
            idx_train_ag[index] = True
    return idx_train_ag, pseudo_labels

def Centroid_train(output, labels, nclass, device):
    centroid_train = torch.tensor([]).to(device)
    for i in range(nclass):
        index = torch.where(labels==i)[0]
        centroid_i = torch.mean(output[index], dim=0)
        centroid_train = torch.cat((centroid_train, centroid_i))
    centroid_train = centroid_train.reshape(-1,nclass)
    return centroid_train

def negative_sampling(idx_train, args, adj):
    train_index = torch.where(idx_train)[0].cpu().tolist()
    positive_index = random.sample(train_index, args.n_positive)
    negative_index = []
    adj = adj.to_dense()
    for index in positive_index:
        indices = torch.where(adj[index] == 0)[0].cpu().tolist()
        negative_index = negative_index + random.sample(indices, args.n_negative_per_positive)
    return positive_index, negative_index



