import pdb
import sys
sys.path.append("../")
sys.path.append(".")
import numpy as np
import torch
from loss import vae_loss as vl
from Data.GM12878_DataModule import GM12878Module

dm = GM12878Module()
dm.setup(stage='fit')
ds = torch.tensor(dm.train_dataloader().dataset.data[0:10])

PARAM_PATH  = "lightning_logs/version_0/hparams.yaml"
WEIGHT_PATH = "lightning_logs/version_0/checkpoints/epoch=264.ckpt" 
vae_loss = vl.VaeLoss(PARAM_PATH, WEIGHT_PATH)
print(vae_loss(ds[0:2],ds[2:4]))
print(vae_loss(ds[0:2],ds[0:2]))
