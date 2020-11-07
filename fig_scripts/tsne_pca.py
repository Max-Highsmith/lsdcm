import pdb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from matplotlib.widgets import Slider
import pdb
import torch
import sys
sys.path.append("Data")
sys.path.append(".")
sys.path.append("../")
import argparse
import numpy as np
from sklearn.decomposition import PCA
import yaml
import glob
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from Data.GM12878_DataModule import GM12878Module
from VAE_Module import VAE_Model

color_list = ['bisque',
        'lightcoral',
        'red',
        'sienna',
        'peru',
        'darkorange',
        'gold',
        'chartreuse',
        'forestgreen',
        'turquoise',
        'teal',
        'dodgerblue',
        'cornflowerblue',
        'navy',
        'slateblue',
        'blueviolet',
        'mediumorchid',
        'thistle',
        'plum',
        'darkslategrey',
        'orchid',
        'deepskyblue',
        'black',
        'silver']


#Get Dataset
parser    = argparse.ArgumentParser()
parser.add_argument("version")
args      = parser.parse_args()

VERSION = args.version
PATH    = glob.glob("lightning_logs/version_"+str(VERSION)+"*/checkpoints/*")[0]
op = open("lightning_logs/version_"+str(VERSION)+"/hparams.yaml")
hparams = yaml.load(op)
print(hparams)

dm_train = GM12878Module()
dm_train.setup(stage='fit')
dm_test = GM12878Module()
dm_test.setup(stage='test')
ds_train         = dm_train.train_dataloader().dataset.target
ds_val           = dm_train.val_dataloader().dataset.target
ds_test          = dm_test.test_dataloader().dataset.target

#Get Model
model   = VAE_Model(
        condensed_latent=hparams['condensed_latent'],
        gamma=['gamma'],
        kld_weight=['kld_weight'],
        latent_dim=hparams['latent_dim'],
        lr=hparams['lr'],
        pre_latent=hparams['pre_latent'])

pretrained_model = model.load_from_checkpoint(PATH)
pretrained_model.freeze()

train_info = dm_train.train_dataloader().dataset.info
val_info   = dm_train.val_dataloader().dataset.info
test_info  = dm_test.test_dataloader().dataset.info


all_outs   = []
all_colors = []

for i in range(0, ds_train.shape[0]):
    test_out = pretrained_model(torch.tensor(ds_train[i:i+1]))
    all_outs.append(test_out[2])
    all_colors.append(color_list[train_info[i]])

for i in range(0, ds_val.shape[0]):
    test_out = pretrained_model(torch.tensor(ds_val[i:i+1]))
    all_outs.append(test_out[2])
    all_colors.append(color_list[val_info[i]])

for i in range(0, ds_test.shape[0]):
    test_out = pretrained_model(torch.tensor(ds_test[i:i+1]))
    all_outs.append(test_out[2])
    all_colors.append(color_list[test_info[i]])

latent_zs = torch.cat(all_outs,0)

pdb.set_trace()

pca     = PCA(n_components=2)
pca_z  = pca.fit_transform(latent_zs)

tsn     = TSNE(n_components=2)
tsne_z  = tsn.fit_transform(latent_zs)


fig, ax = plt.subplots()
ax.scatter(pca_z[:,0],
            pca_z[:,1],
            c=all_colors,
            s=[10]*len(pca_z))

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("PC1 ({:.2f}%)".format(100*pca.explained_variance_ratio_[0])+")")
ax.set_ylabel("PC2 ({:.2f}%)".format(100*pca.explained_variance_ratio_[1])+")")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()

fig, ax = plt.subplots()
ax.scatter(tsne_z[:,0],
            tsne_z[:,1],
            c=all_colors,
            s=[10]*len(tsne_z))
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("Dim 1")
ax.set_ylabel("Dim 2")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()
