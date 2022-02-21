# DR-GST

This repo is for source code of WWW 2022 paper "Confidence May Cheat: Self-Training on Graph Neural Networks under Distribution Shift".

Paper Link: https://arxiv.org/abs/2201.11349

## Environment

- python == 3.8.8
- pytorch == 1.8.1
- dgl -cuda11.1 == 0.6.1
- networkx == 2.5
- numpy == 1.20.2

## Main Methods

```python
python main.py --dataset dataset --m model --labelrate labelrate --s stage --t threshold --b beta --drop_method drop_method --droprate droprate
```

- **dataset:** including [Cora, Citeseer, Pubmed, CoraFull, Flickr]
- **model:** including [GCN, GAT, APPNP]
- **labelrate:** including [3, 5, 10, 20]
- **stage:** ranging from 1 to 10
- **drop_method:** including [dropout, dropedge]

## Part of Parameters

### labelrate = 20, drop_method = dropout

| Dataset-labelrate-threshold-beta-droprate      |         |
| ---------------------------------------------- | ------- |
| Cora-20-0.530000-0.333333-0.500000-dropout     | stage=2 |
| Citeseer-20-0.700000-0.500000-0.300000-dropout | stage=9 |
| Pubmed-20-0.740000-1.000000-0.500000-dropout   | stage=2 |
| CoraFull-20-0.800000-0.250000-0.400000-dropout | stage=5 |
| Flickr-20-0.890000-1.000000-0.100000-dropout   | stage=9 |

### labelrate = 20, drop_method = dropedge

| Dataset-labelrate-threshold-beta-droprate       |           |
| ----------------------------------------------- | --------- |
| Cora-20-0.540000-0.500000-0.400000-dropedge     | stage = 5 |
| Citeseer-20-0.600000-0.500000-0.100000-dropedge | stage = 3 |
| Pubmed-20-0.710000-1.000000-0.500000-dropedge   | stage = 2 |
| CoraFull-20-0.910000-0.200000-0.500000-dropedge | stage = 9 |
| Flickr-20-0.950000-0.500000-0.200000-dropedge   | stage = 6 |

### labelrate = 10, drop_method = dropout

| Dataset-labelrate-threshold-beta-droprate      |         |
| ---------------------------------------------- | ------- |
| Cora-10-0.700000-0.333333-0.300000-dropout     | stage=6 |
| Citeseer-10-0.550000-0.500000-0.300000-dropout | stage=4 |
| Pubmed-10-0.750000-1.000000-0.300000-dropout   | stage=1 |
| CoraFull-10-0.780000-0.333333-0.100000-dropout | stage=6 |
| Flickr-10-0.960000-1.000000-0.100000-dropout   | stage=7 |

### labelrate = 10, drop_method = dropedge

| Dataset-labelrate-threshold-beta-droprate       |           |
| ----------------------------------------------- | --------- |
| Cora-10-0.650000-0.500000-0.500000-dropedge     | stage = 3 |
| Citeseer-10-0.650000-0.500000-0.300000-dropedge | stage = 6 |
| Pubmed-10-0.600000-0.666667-0.300000-dropedge   | stage = 1 |
| CoraFull-10-0.630000-0.333333-0.300000-dropedge | stage = 2 |
| Flickr-10-0.930000-0.666667-0.100000-dropedge   | stage = 2 |

### labelrate = 5, drop_method = dropout

| Dataset-labelrate-threshold-beta-droprate     |         |
| --------------------------------------------- | ------- |
| Cora-5-0.600000-0.500000-0.500000-dropout     | stage=2 |
| Citeseer-5-0.450000-0.666667-0.500000-dropout | stage=9 |
| Pubmed-5-0.870000-1.000000-0.300000-dropout   | stage=1 |
| CoraFull-5-0.690000-0.333333-0.100000-dropout | stage=2 |
| Flickr-5-0.930000-1.000000-0.100000-dropout   | stage=2 |

### labelrate = 5, drop_method = dropedge

| Dataset-labelrate-threshold-beta-droprate      |           |
| ---------------------------------------------- | --------- |
| Cora-5-0.600000-0.500000-0.300000-dropedge     | stage = 2 |
| Citeseer-5-0.400000-0.500000-0.100000-dropedge | stage = 3 |
| Pubmed-5-0.600000-1.000000-0.300000-dropedge   | stage = 1 |
| CoraFull-5-0.600000-0.500000-0.300000-dropedge | stage = 2 |
| Flickr-5-0.960000-1.000000-0.500000-dropedge   | stage = 4 |

### labelrate = 3, drop_method = dropout

| Dataset-labelrate-threshold-beta-droprate     |         |
| --------------------------------------------- | ------- |
| Cora-3-0.700000-0.500000-0.300000-dropout     | stage=3 |
| Citeseer-3-0.450000-0.666667-0.300000-dropout | stage=2 |
| Pubmed-3-0.780000-0.500000-0.500000-dropout   | stage=1 |
| CoraFull-3-0.870000-0.333333-0.300000-dropout | stage=2 |
| Flickr-3-0.960000-1.000000-0.500000-dropout   | stage=4 |

### labelrate = 3, drop_method = dropedge

| Dataset-labelrate-threshold-beta-droprate      |           |
| ---------------------------------------------- | --------- |
| Cora-3-0.400000-0.333333-0.100000-dropedge     | stage = 2 |
| Citeseer-3-0.450000-0.333333-0.500000-dropedge | stage = 1 |
| Pubmed-3-0.810000-0.500000-0.300000-dropedge   | stage = 1 |
| CoraFull-3-0.660000-0.500000-0.300000-dropedge | stage = 1 |
| Flickr-3-0.900000-0.666667-0.500000-dropedge   | stage = 1 |

## Contact

If you have any questions, please feel free to contact me with [liuhongrui@bupt.edu.cn](mailto:liuhongrui@bupt.edu.cn)

