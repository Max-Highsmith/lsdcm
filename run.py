from pytorch_lightning import Trainer
from Data.GM12878_DataModule import GM12878Module
from AE_Module import AE_Model
from VAE_Module import VAE_Model
dm      = GM12878Module()
#model   = AE_Model()
model    = VAE_Model()
trainer = Trainer(gpus=1)
trainer.fit(model, dm)


