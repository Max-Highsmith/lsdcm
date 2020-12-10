import sys
sys.path.append(".")
sys.path.append("../")
import vision_metrics as vm
import numpy as np
import glob
import yaml
from Data.GM12878_DataModule import GM12878Module
import matplotlib.pyplot as plt
import pdb
import torch
import numpy

#vae
from VAE_Module import VAE_Model

#load data
'''
dm_test = GM12878Module(batch_size=1, res=10000, piece_size=269)
dm_test.prepare_data()
#dm_test.setup(stage='test')
dm_test.setup(stage=20)
ds     = torch.from_numpy(dm_test.test_dataloader().dataset.data[9:10])
target = torch.from_numpy(dm_test.test_dataloader().dataset.target[9:10])
'''

#
VERSION  = 4 
PATH     = glob.glob("lightning_logs/version_"+str(VERSION)+"*/checkpoints/*")[0]
op       = open("lightning_logs/version_"+str(VERSION)+"/hparams.yaml")
hparams  = yaml.load(op)
pdb.set_trace()
model = VAE_Model(
        condensed_latent=hparams['condensed_latent'],
        gamma=['gamma'],
        kld_weight=['kld_weight'],
        latent_dim=hparams['latent_dim'],
        lr=hparams['lr'],
        pre_latent=hparams['pre_latent'])

vae_model = model.load_from_checkpoint(PATH)


#deepchromap_out = vae_model(ds).detach()[0][0]
#lowres_out  = ds[0][0][6:-6,6:-6]
#target_out  = target[0][0][6:-6,6:-6]



v_m ={}
chro = 20
#compute vision metrics
print("vae")
visionMetrics = vm.VisionMetrics()
visionMetrics.setDataset(chro)
v_m[chro, 'vae']=visionMetrics.getMetrics(model=vae_model, spliter="vae")
 
model_names  = ['downsampled', 'vae']
metric_names = ['pcc','spc','ssim', 'mse', 'snr']

cell_text = []

for mod_nm in model_names:
    met_list = []
    for met_nm in metric_names:
        if mod_nm=="downsampled":
            met_list.append("{:.4f}".format(np.mean(v_m[chro, "vae"]['pre_'+str(met_nm)])))
        else:
            met_list.append("{:.4f}".format(np.mean(v_m[chro, mod_nm]['pas_'+str(met_nm)])))
    cell_text.append(met_list)

plt.subplots_adjust(left=0.2, top=0.8)
plt.table(cellText=cell_text, rowLabels=model_names, colLabels=metric_names, loc='top')
plt.title(chro)
plt.show()
pdb.set_trace()



