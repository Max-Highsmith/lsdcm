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

parser    = argparse.ArgumentParser()
parser.add_argument("version")
args      = parser.parse_args()

VERSION = args.version
PATH    = glob.glob("lightning_logs/version_"+str(VERSION)+"*/checkpoints/*")[0]
op = open("lightning_logs/version_"+str(VERSION)+"/hparams.yaml")
hparams = yaml.load(op)
print(hparams)

dm_train = GM12878Module()
dm_train.setup(stage='test')
ds        = dm_train.test_dataloader().dataset.data


model   = VAE_Model(
        condensed_latent=hparams['condensed_latent'],
        gamma=['gamma'],
        kld_weight=['kld_weight'],
        latent_dim=hparams['latent_dim'],
        lr=hparams['lr'],
        pre_latent=hparams['pre_latent'])

pretrained_model = model.load_from_checkpoint(PATH)
pretrained_model.freeze()


#latent dim PCA
fig, ax  = plt.subplots(2)
real_z   = pretrained_model.get_z(torch.tensor(ds))
pca      = PCA(n_components=2)
cond_z   = pca.fit_transform(real_z[0])
real_scatter = ax[0].scatter(cond_z[:,0], 
                        cond_z[:,1],
                        color=["green"]*len(cond_z),
                        picker=5,
                        s=[10]*len(cond_z))
ax[0].set_xlabel("PC1 ("+"{:.2f}".format(pca.explained_variance_ratio_[0])+")")
ax[0].set_ylabel("PC2 ("+"{:.2f}".format(pca.explained_variance_ratio_[1])+")")
ax[0].set_xticks([0])
ax[0].set_yticks([0])

def on_pick(event):
    real_scatter._facecolors[event.ind,:] = (1,0,0,1)
    real_scatter._edgecolors[event.ind,:] = (1,0,0,1)
    fig.canvas.draw()

def on_click(event):
    print("X:"+str(event.xdata)+"  Y:"+str(event.ydata))
    full_loc = pca.inverse_transform([[event.xdata, event.ydata]])
    out_im   = pretrained_model.decode(torch.from_numpy(full_loc).reshape(1,1,1,hparams['latent_dim']).type(torch.float32))
    ax[1].imshow(out_im[0][0], cmap="Reds")
    fig.canvas.flush_events()
    fig.canvas.draw()


fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()

random_z    = torch.normal(mean=torch.zeros(100, hparams['latent_dim']), std=(1+torch.ones(100, hparams['latent_dim'])))
cond_rand_z = pca.transform(random_z) 
rand_scatter = ax.scatter(cond_rand_z[:,0], cond_rand_z[:,1])
plt.show()

gen_x    = pretrained_model.decode(random_z)
fig, ax  = plt.subplots(5)

for i in range(0,5):
        ax[i].imshow(gen_x[i][0], cmap="Oranges")
plt.show()



fig, ax = plt.subplots(1,5)
for i, val in enumerate([-2,-1.5,-1,-.5,0]):
    inc_line = torch.bernoulli(torch.rand(200)) *val
    ax[i].imshow(pretrained_model.decode(torch.zeros(1,200)+inc_line)[0][0], cmap="Oranges")

plt.show()








