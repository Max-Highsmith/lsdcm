import argparse
import sys
sys.path.append("Data")
sys.path.append(".")
sys.path.append("../")
import numpy as np
from sklearn.decomposition import PCA
import glob
import yaml
import matplotlib.pyplot as plt
import torch
import pdb
from pytorch_lightning import Trainer
from Data.GM12878_DataModule import GM12878Module
from GAN_Module import GAN_Model

parser = argparse.ArgumentParser()
parser.add_argument("version")
args  =  parser.parse_args()

VERSION = args.version
PATH    = glob.glob("lightning_logs/version_"+str(VERSION)+"*/checkpoints/*")[0]
op = open("lightning_logs/version_"+str(VERSION)+"/hparams.yaml")
hparams = yaml.load(op)
print(hparams)
dm_train      = GM12878Module(batch_size=2)
dm_train.prepare_data()
dm_train.setup(stage='fit')

model   = GAN_Model()

pretrained_model = model.load_from_checkpoint(PATH)
pretrained_model.freeze()


for epoch in dm_train.train_dataloader():
    data, target, info = epoch
    output             = pretrained_model(target)
    fig, ax = plt.subplots(3)
    ax[0].imshow(output[0][0], cmap="Reds")
    ax[0].set_title("output")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].imshow(target[0][0], cmap="Reds")
    ax[1].set_title("target")
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].imshow(data[0][0], cmap="Reds")
    ax[2].set_title("data")
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    plt.show()
    pdb.set_trace()

