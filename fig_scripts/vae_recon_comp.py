import pdb
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
test_out = pretrained_model(torch.tensor(ds_train[0:10]))
#Show Comparision
diff_out = test_out[0].detach()-test_out[1].detach()
fig, ax = plt.subplots(2,3)
for i in range(0,2):
    origin_im = ax[i,0].imshow(test_out[1][i][0].detach(), cmap="Reds")
    recon_im  = ax[i,1].imshow(test_out[0][i][0].detach(), cmap="Reds")
    comp_im   = ax[i,2].imshow(diff_out[i][0], cmap="RdBu", vmin=-1, vmax=1)
    ax[i,0].set_xticks([])
    ax[i,0].set_yticks([])
    ax[i,1].set_xticks([])
    ax[i,1].set_yticks([])
    ax[i,2].set_xticks([])
    ax[i,2].set_yticks([])
fig.colorbar(comp_im, orientation="horizontal")
plt.show()


#TODO graphs showing genomic distance


