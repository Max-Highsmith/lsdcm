import argparse
import numpy as np
from sklearn.decomposition import PCA
import glob
import yaml
import matplotlib.pyplot as plt
import torch
import pdb
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

#train
x             = torch.from_numpy(dm_train.train_dataloader().dataset.target[30:60])
mu, log_var   = pretrained_model.encode(x)
z      = pretrained_model.reparameterize(mu, log_var)
dec_x  = pretrained_model.decode(z)
output = pretrained_model(x)
fig, ax = plt.subplots(3)
for i in range(0,mu.shape[0]):
    ax[0].plot(mu[i])
    ax[1].plot(log_var[i])
    ax[2].plot(z[i])
ax[1].set_xlabel("Latent Node Representation")
ax[0].set_ylabel("Mu")
ax[1].set_ylabel("Log Var")
ax[2].set_ylabel("Z")
ax[0].set_title("train")
plt.show()
fig, ax = plt.subplots(2,3)
for i in range(0,3):
    ax[0,i].imshow(x[i*3][0], cmap="Reds")
    ax[1,i].imshow(output[0][i*3][0], cmap="Reds")
plt.show()

#test
x             = torch.from_numpy(dm_test.test_dataloader().dataset.target[0:256])
mu, log_var   = pretrained_model.encode(x)
z      = pretrained_model.reparameterize(mu, log_var)
dec_x  = pretrained_model.decode(z)
output = pretrained_model(x)
fig, ax = plt.subplots(3)
for i in range(0,mu.shape[0]):
    ax[0].plot(mu[i])
    ax[1].plot(log_var[i])
    ax[2].plot(z[i])
ax[1].set_xlabel("Latent Node Representation")
ax[0].set_ylabel("Mu")
ax[1].set_ylabel("Log Var")
ax[2].set_ylabel("Z")
ax[0].set_title("test")
plt.show()
fig, ax = plt.subplots(2,3)
for i in range(0,3):
    ax[0,i].imshow(x[i*3][0], cmap="Greens")
    ax[1,i].imshow(output[0][i*3][0], cmap="Greens")
plt.show()

#latent dim PCA
pca    = PCA(n_components=2)
cond_z = pca.fit_transform(z)
z_hat  = pca.inverse_transform(cond_z)
plt.scatter(cond_z[:,0], cond_z[:,1])
plt.show()

#generate random z
random_z = torch.normal(mean=torch.zeros(30,hparams['latent_dim']), std=torch.ones(30,hparams['latent_dim']))
gen_x    = pretrained_model.decode(random_z)
fig, ax  = plt.subplots(1,3)
for i in range(0,3):
    ax[i].imshow(gen_x[i][0], cmap="Oranges")
plt.show()

#random using PCA
random_pca   = torch.normal(mean=torch.zeros(3,2), std=torch.ones(3,2))
random_pca_z = torch.tensor(pca.inverse_transform(random_pca))
gen_pca_x    = pretrained_model.decode(random_pca_z.type(torch.float32))
fig, ax      = plt.subplots(1,3)
for i in range(0,3):
    ax[i].imshow(gen_pca_x[i][0], cmap="Blues")

plt.show()

for i in range(0,2):
    plt.plot(z[i], color="red")
    plt.plot(random_z[i], color="purple")

plt.show()
pdb.set_trace()
