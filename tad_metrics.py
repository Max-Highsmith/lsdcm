from loss import insulation as ins
import matplotlib.pyplot as plt
import glob
import yaml
import subprocess
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Data.GM12878_DataModule import GM12878Module
from Data.K562_DataModule    import K562Module

#Load Models
import GAN_Module as deepchromap
import models_to_compare.models.deephic as deephic
import models_to_compare.models.hicplus as hicplus
import models_to_compare.models.hicsr   as hicsr

methods = ['downs', 'hicplus', 'deephic', 'hicsr', 'vehicle'] 
colors  = ['black', 'silver', 'blue', 'darkviolet', 'coral']

getIns   = ins.computeInsulation()

##VeHICLE
#VERSION  = 8
#PATH     = glob.glob("lightning_logs/version_"+str(VERSION)+"*/checkpoints/*")[0]
#op       = open("lightning_logs/version_"+str(VERSION)+"/hparams.yaml")
#hparams  = yaml.load(op)
WEIGHT_PATH="deepchromap_weights.ckpt"
deepChroModel = deepchromap.GAN_Model()
model_deepchromap = deepChroModel.load_from_checkpoint(WEIGHT_PATH)

##HiCPlus
model_hicplus = hicplus.Net(40,28)
model_hicplus.load_state_dict(torch.load("models_to_compare/weights/pytorch_HindIII_model_40000"))

##HiCSR
model_hicsr   = hicsr.Generator(num_res_blocks=15)
HICSR_WEIGHTS = "models_to_compare/weights/test_HiCSR_50.pth"
model_hicsr.load_state_dict(torch.load(HICSR_WEIGHTS))
model_hicsr.eval()

#DeepHiC
model_deephic = deephic.Generator(scale_factor=1, in_channel=1, resblock_num=5)
model_deephic.load_state_dict(torch.load("models_to_compare/weights/deephic_200.pytorch"))


#CHRO       = 4
RES        = 10000
PIECE_SIZE = 269

CELL_LINE = "K562"
if CELL_LINE == "GM12878":
    dm_test = GM12878Module(batch_size=1, res=RES, piece_size=PIECE_SIZE)
    dm_test.prepare_data()
    dm_test.setup(stage='test')

if CELL_LINE == "K562":
    dm_test = K562Module(batch_size=1, res=RES, piece_size=PIECE_SIZE)
    dm_test.prepare_data()
    dm_test.setup(stage='test')


full_insulation_dist = {
        'hicsr':[],
        'down':[],
        'vehicle':[],
        'deephic':[],
        'hicplus':[]
        }

directionality_comp = {
        'hicsr':[],
        'down':[],
        'vehicle':[],
        'deephic':[],
        'hicplus':[],
        'target':[]
        }


def getTadBorderDists(x,y):
    nearest_distances = []
    for border1 in x:
        if border1 >50 and border1 <101:
            nearest = 9999
            for border2 in y:
                dist = abs(border1-border2)
                if dist < nearest:
                    nearest = dist
            nearest_distances.append(nearest)

    return nearest_distances



STEP_SIZE = 50
BUFF_SIZE = 36

NUM_ITEMS = dm_test.test_dataloader().dataset.data.shape[0]
for s, sample in enumerate(dm_test.test_dataloader()):
    print(str(s)+"/"+str(NUM_ITEMS))
    data, target, _ = sample
    downs   = data[0][0]
    target  = target[0][0]
    
    #Pass through Models
    #Pass through HicPlus
    hicplus_out = torch.zeros((PIECE_SIZE, PIECE_SIZE))
    for i in range(0, PIECE_SIZE-40, 28):
        for j in range(0, PIECE_SIZE-40, 28):
            temp                            = data[:,:,i:i+40, j:j+40]
            hicplus_out[i+6:i+34, j+6:j+34] =  model_hicplus(temp)
    hicplus_out = hicplus_out.detach()[6:-6, 6:-6]

    #Pass through Deephic
    deephic_out = torch.zeros((PIECE_SIZE, PIECE_SIZE))
    for i in range(0, PIECE_SIZE-40, 28):
        for j in range(0, PIECE_SIZE -40, 28):
            temp                            = data[:,:,i:i+40, j:j+40]
            deephic_out[i+6:i+34, j+6:j+34] = model_deephic(temp)[:,:,6:34, 6:34]
    deephic_out = deephic_out.detach()[6:-6,6:-6]

    #Pass through HiCSR
    hicsr_out = torch.zeros((PIECE_SIZE, PIECE_SIZE))
    for i in range(0, PIECE_SIZE-40, 28):
        for j in range(0, PIECE_SIZE-40, 28):
            temp                          = data[:,:,i:i+40, j:j+40]
            hicsr_out[i+6:i+34, j+6:j+34] = model_hicsr(temp)
    hicsr_out = hicsr_out.detach()[6:-6, 6:-6]
    hicsr_out = torch.clamp(hicsr_out,0, 100000000)

    #PASS through VeHICLE TODO TODO
    vehicle_out = model_deepchromap(data).detach()[0][0]


    downs   = downs[6:-6,6:-6]
    target  = target[6:-6,6:-6]

    
    down_tads      = getIns.forward(downs.reshape(1,1,257,257))[2][0].tolist()
    target_tads    = getIns.forward(target.reshape(1,1,257,257))[2][0].tolist()
    vehicle_tads   = getIns.forward(vehicle_out.reshape(1,1,257,257))[2][0].tolist()
    hicsr_tads     = getIns.forward(hicsr_out.reshape(1,1,257,257))[2][0].tolist()
    deephic_tads   = getIns.forward(deephic_out.reshape(1,1,257,257))[2][0].tolist()
    hicplus_tads   = getIns.forward(hicplus_out.reshape(1,1,257,257))[2][0].tolist()
    

    temp_down    = np.array(getTadBorderDists(target_tads, down_tads))
    temp_hicsr   = np.array(getTadBorderDists(target_tads, hicsr_tads))
    temp_deephic = np.array(getTadBorderDists(target_tads, deephic_tads))
    temp_hicplus = np.array(getTadBorderDists(target_tads, hicplus_tads))
    temp_vehicle = np.array(getTadBorderDists(target_tads, vehicle_tads))

    THRESHOLD = 8
    full_insulation_dist['down'].extend(temp_down[temp_down<THRESHOLD].tolist())
    full_insulation_dist['hicsr'].extend(temp_hicsr[temp_hicsr<THRESHOLD].tolist())
    full_insulation_dist['deephic'].extend(temp_deephic[temp_deephic<THRESHOLD].tolist())
    full_insulation_dist['hicplus'].extend(temp_hicplus[temp_hicplus<THRESHOLD].tolist())
    full_insulation_dist['vehicle'].extend(temp_vehicle[temp_vehicle<THRESHOLD].tolist())


    #SHow images
    '''
    fig, ax = plt.subplots(3)
    ax[0].imshow(hicsr_out, cmap="Reds")
    ax[1].imshow(vehicle_out, cmap="Reds")
    ax[2].imshow(target, cmap="Reds")
    plt.show()
    '''
    '''
    fig, ax = plt.subplots(2)
    ax[0].plot(getIns.forward(downs.reshape(1,1,257,257))[0][0][0], label='downs' ,color='black')
    ax[0].plot(getIns.forward(hicplus_out.reshape(1,1,257,257))[0][0][0], label="hicplus", color='silver')
    ax[0].plot(getIns.forward(deephic_out.reshape(1,1,257,257))[0][0][0], label="hicout", color='blue')
    ax[0].plot(getIns.forward(hicsr_out.reshape(1,1,257,257))[0][0][0], label="hicsr", color='darkviolet')
    ax[0].plot(getIns.forward(vehicle_out.reshape(1,1,257,257))[0][0][0], label='vehicle' ,color='coral')
    ax[0].plot(getIns.forward(target.reshape(1,1,257,257))[0][0][0], label='target' ,color='cyan')
    ax[1].plot(getIns.forward(downs.reshape(1,1,257,257))[1][0][0], label='downs' ,color='black')
    ax[1].plot(getIns.forward(hicplus_out.reshape(1,1,257,257))[1][0][0], label="hicplus", color='silver')
    ax[1].plot(getIns.forward(deephic_out.reshape(1,1,257,257))[1][0][0], label="deephic", color='blue')
    ax[1].plot(getIns.forward(hicsr_out.reshape(1,1,257,257))[1][0][0], label='hicsr' ,color='darkviolet')
    ax[1].plot(getIns.forward(vehicle_out.reshape(1,1,257,257))[1][0][0], label='vehicle' ,color='coral')
    ax[1].plot(getIns.forward(target.reshape(1,1,257,257))[1][0][0], label='target' ,color='cyan')
    ax[0].set_xticks(list(range(s*50, (s*50)+269,50)))
    ax[0].spines['right'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    plt.legend()
    plt.show()
    pdb.set_trace()
    '''
    directionality_comp['down'].extend(getIns.forward(downs.reshape(1,1,257,257))[1][0][0][0:50].tolist())
    directionality_comp['hicsr'].extend(getIns.forward(hicsr_out.reshape(1,1,257,257))[1][0][0][0:50].tolist())
    directionality_comp['hicplus'].extend(getIns.forward(hicplus_out.reshape(1,1,257,257))[1][0][0][0:50].tolist())
    directionality_comp['deephic'].extend(getIns.forward(deephic_out.reshape(1,1,257,257))[1][0][0][0:50].tolist())
    directionality_comp['vehicle'].extend(getIns.forward(vehicle_out.reshape(1,1,257,257))[1][0][0][0:50].tolist())
    directionality_comp['target'].extend(getIns.forward(target.reshape(1,1,257,257))[1][0][0][0:50].tolist())

    if s == 20:
        pdb.set_trace()
        fig, ax = plt.subplots(1)
        box_data = [
                np.array(full_insulation_dist['down'], ),
                full_insulation_dist['hicplus'],
                full_insulation_dist['deephic'],
                full_insulation_dist['hicsr'],
                full_insulation_dist['vehicle']]
        positions = [1,2,3,4,5]
        bp = ax.boxplot(box_data,
                positions=positions,
                showfliers=False,
                patch_artist=True,
                showmeans=True,
                notch=True)

        ax.set_xticklabels(['down', 'hicplus', 'deephic', 'hicsr', 'vehicle'])
        ax.set_ylabel("TAD Border Distance (10kb)")
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        for patch, color in zip(bp['medians'], colors):
            patch.set_c('white')
        plt.show()

'''   
fig, ax = plt.subplots(1)
box_data = [full_insulation_dist['down'],
        full_insulation_dist['hicplus'],
        full_insulation_dist['deephic'],
        full_insulation_dist['hicsr'],
        full_insulation_dist['vehicle']]
positions = [1,2,3,4,5]
bp = ax.boxplot(box_data,
        positions=positions,
        showfliers=False,
        patch_artist=True,
        notch=True)
ax.set_xticklabels(['down', 'hicplus', 'deephic', 'hicsr', 'vehicle'])
ax.set_ylabel("TAD Border Distance (10kb)")
pdb.set_trace()
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
for patch, color in zip(bp['medians'], colors):
    patch.set_c('white')
plt.show()
'''
