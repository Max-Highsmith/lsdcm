import matplotlib.pyplot as plt
import torch
import pdb
from pytorch_lightning import Trainer
from Data.GM12878_DataModule import GM12878Module
from VAE_Module import VAE_Model

dm      = GM12878Module(batch_size=80)
dm.setup(stage='test')
model   = VAE_Model()
PATH     = "lightning_logs/version_5/checkpoints/epoch=5.ckpt"
pretrained_model = model.load_from_checkpoint(PATH)
pretrained_model.freeze()

x             = torch.from_numpy(dm.test_dataloader().dataset.target[0:30])
mu, log_var   = pretrained_model.encode(x)


#plot mu and
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
plt.show()
fig, ax = plt.subplots(2,3)
for i in range(0,3):
    ax[0,i].imshow(x[i*3][0])
    ax[1,i].imshow(output[0][i*3][0])
plt.show()

#trainer = Trainer(gpus=1)
#trainer.fit(model, dm)
#trainer.test(model,dm)



