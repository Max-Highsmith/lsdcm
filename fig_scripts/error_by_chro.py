import torch.nn.functional as F
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

parser = argparse.ArgumentParser()
parser.add_argument("version")
args  =  parser.parse_args()

VERSION = args.version
PATH    = glob.glob("lightning_logs/version_"+str(VERSION)+"*/checkpoints/*")[0]
op = open("lightning_logs/version_"+str(VERSION)+"/hparams.yaml")
hparams = yaml.load(op)
print(hparams)
dm_train      = GM12878Module(batch_size=80)
dm_train.setup(stage='fit')
dm_test      = GM12878Module(batch_size=80)
dm_test.setup(stage='test')

model   = VAE_Model(
        condensed_latent=hparams['condensed_latent'],
        gamma=['gamma'],
        kld_weight=['kld_weight'],
        latent_dim=hparams['latent_dim'],
        lr=hparams['lr'],
        pre_latent=hparams['pre_latent'])

pretrained_model = model.load_from_checkpoint(PATH)
pretrained_model.freeze()

chro    = 99
mse     = {}
in_freq = {}
window_inc = 5
idx        = 0
for i, batch in enumerate(dm_test.test_dataloader()):
    data, target = batch
    output = pretrained_model(target)
    for s in range(0, output[0].shape[0]):
        print("idx:"+str(idx))
        idx = idx +1
        if idx > 0:
            chro = 4
        if idx > 369:
            chro = 14
        if idx > 369+170:
            chro = 16
        if idx>369+170+145:
            chro = 20
        if idx>369+170+145+114:
            pdb.set_trace()
        for i in range(0, output[0].shape[3]-40, window_inc):
            for j in range(0, output[0].shape[3]-40,window_inc): 
                genomic_dist = abs(i-j)
                if (chro, genomic_dist) not in mse.keys():
                    mse[chro, genomic_dist]     = []
                    in_freq["target",genomic_dist] = []
                    in_freq["recon",genomic_dist] = []
                mse[chro, genomic_dist].append(torch.nn.functional.mse_loss(output[0][s][0][i:i+40, j:j+40], target[s][0][i:i+40, j:j+40]).item())
                in_freq["target",genomic_dist].append(torch.sum(target[s][0][i:i+40, j:j+40]).item())
                in_freq["recon",genomic_dist].append(torch.sum(output[0][s][0][i:i+40, j:j+40]).item())

mse_lines = {}
for chro in [4,14,16,20]:
    for i in range(0, output[0].shape[3]-40,  window_inc):
        if chro not in mse_lines:
            mse_lines[chro] = []
        mse_lines[chro].append(np.mean(mse[chro, i]))

fig, ax = plt.subplots()
for chro in [4,14,16,20]:
    ax.plot(mse_lines[chro], label="chro:"+str(chro))
ax.set_xticks(np.array([0,40,80,120,160,200])/window_inc)
ax.set_xticklabels([0,40,80,120,160,200])
ax.legend()
plt.show()

#contact count boxplots:
data_to_plot = []
fig, ax = plt.subplots()
for gd in [0,40,80,120,160,200]:
    for ty in ['target', 'recon']:
        data_to_plot.append(in_freq[ty,gd])

bp = ax.boxplot(data_to_plot, patch_artist=True)
for b, box in enumerate(bp['boxes']):
    if b %2 ==0:
        box.set(facecolor = 'red')
    else:
        box.set(facecolor = 'blue')
ax.set_xticks([1.5,3.5,5.5,7.5,9.5,11.5])
ax.set_xticklabels(["0","40","80","120","160","200"])
ax.set_yscale('log', basey=2)
ax.set_xlabel("Genomic Distance (10kb)")
ax.set_ylabel("(Interaction Frequency)")
plt.show()

#contact mean squared






