import matplotlib.pyplot as plt
import torch
import pdb
from pytorch_lightning import Trainer
from Data.GM12878_DataModule import GM12878Module
from VAE_Module import VAE_Model

dm      = GM12878Module()
dm.setup(stage='test')
model   = VAE_Model()
#PATH    = "lightning_logs/version_3/checkpoints/epoch=999.ckpt"a
PATH     = "lightning_logs/version_28/checkpoints/epoch=755.ckpt"
pretrained_model = model.load_from_checkpoint(PATH)
pretrained_model.freeze()

x      = torch.from_numpy(dm.test_dataloader().dataset.target[0:1])
output = pretrained_model(x)
fig, ax = plt.subplots(2)
ax[0].imshow(x[0][0])
ax[1].imshow(output[0][0][0])
plt.show()

#trainer = Trainer(gpus=1)
#trainer.fit(model, dm)
#trainer.test(model,dm)



