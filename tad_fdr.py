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
from Data.IMR90_DataModule   import IMR90Module
from Data.HMEC_DataModule    import HMECModule

#Load Models
import GAN_Module as deepchromap
import models_to_compare.models.deephic as deephic
import models_to_compare.models.hicplus as hicplus
import models_to_compare.models.hicsr   as hicsr

RES        = 10000
PIECE_SIZE = 269

METHODS = ['down', 'hicplus', 'deephic', 'hicsr', 'vehicle', 'target'] 
METHOD_COLORS  = ['black', 'silver', 'blue', 'darkviolet', 'coral', 'gold']


CELL_LINES       = ['IMR90', 'HMEC', 'GM12878', 'K562']
CELL_LINE_SHAPES = ["o","v","P","*"] 

CHROS            = [20]#,16,14,4]
CHRO_SIZES       = [10,50,100,150]

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



TAD_THRESHOLD = 2
def checkIfCaught(x,y):
    False_Discovery   = 0
    Correct_Discovery = 0
    Missed_Discovery  = 0

    #print("----")
    #print(x)
    #print(y)
    for b, border1 in enumerate(x):
        if border1 >50 and border1 <101:
            nearest = 9999
            for border2 in y:
                dist = abs(border1-border2)
                if dist < nearest:
                    nearest = dist
            #print(str(border1)+":"+str(nearest)+":"+str(TAD_THRESHOLD))
            if nearest < TAD_THRESHOLD:
                Correct_Discovery = Correct_Discovery + 1
            else:
                Missed_Discovery = Missed_Discovery + 1
                
    for b, border1 in enumerate(y):
        if border1 > 50 and border1 < 101:
            nearest = 999
            for border2 in x:
                dist = abs(border1-border2)
                if dist < nearest:
                    nearest = dist
            if nearest < TAD_THRESHOLD:
                1+1
            else:
                False_Discovery = False_Discovery + 1

    return Correct_Discovery, Missed_Discovery, False_Discovery

def getMetrics(correct, missed, falsed):
    precision = correct/(correct+falsed)
    recall    = correct/(correct+missed)
    f1        = 2 * (precision*recall)/(precision+recall)
    return precision, recall, f1

def getModule(cellline):
        if cellline == "HMEC":
            dm_test = HMECModule(batch_size=1, res=RES, piece_size=PIECE_SIZE)

        if cellline == "IMR90":
            dm_test = IMR90Module(batch_size=1, res=RES, piece_size=PIECE_SIZE)

        if cellline == "GM12878":
            dm_test = GM12878Module(batch_size=1, res=RES, piece_size=PIECE_SIZE)

        if cellline == "K562":
            dm_test = K562Module(batch_size=1, res=RES, piece_size=PIECE_SIZE)
        dm_test.prepare_data()
        dm_test.setup(stage=chro)
        return dm_test

def setupFDR():
    fdr =   {
            ('deephic', 'correct'):0,
            ('deephic', 'missed'):0,
            ('deephic', 'false'):0,
            ('down', 'correct'):0,
            ('down', 'missed'):0,
            ('down', 'false'):0,
            ('hicplus', 'correct'):0,
            ('hicplus', 'missed'):0,
            ('hicplus', 'false'):0,
            ('hicsr', 'correct'):0,
            ('hicsr', 'missed'):0,
            ('hicsr', 'false'):0,
            ('vehicle', 'correct'):0,
            ('vehicle', 'missed'):0,
            ('vehicle', 'false'):0,
            ('target', 'correct'):0,
            ('target', 'missed'):0,
            ('target', 'false'):0,
            }   
    return fdr

final_scores = {}

#Compute FDR
for cellline in CELL_LINES: #["GM12878", "K562"]: # "IMR90", "HMEC"]:
    for chro in CHROS: #[20, 16]: #, 14, 4]:
        dm_test = getModule(cellline)
        fdr = setupFDR()
        NUM_ITEMS = dm_test.test_dataloader().dataset.data.shape[0]
        for s, sample in enumerate(dm_test.test_dataloader()):
            #if s >2:
            #    break
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

            temp_downs    = getIns.forward(downs.reshape(1,1,257,257))[2][0].tolist()
            temp_hicplus  = getIns.forward(hicplus_out.reshape(1,1,257,257))[2][0].tolist()
            temp_deephic  = getIns.forward(deephic_out.reshape(1,1,257,257))[2][0].tolist()
            temp_hicsr    = getIns.forward(hicsr_out.reshape(1,1,257,257))[2][0].tolist()
            temp_vehicle  = getIns.forward(vehicle_out.reshape(1,1,257,257))[2][0].tolist()
            temp_target   = getIns.forward(target.reshape(1,1,257,257))[2][0].tolist()
            
                
            temps = [temp_downs, temp_hicplus, temp_deephic, temp_hicsr, temp_vehicle, temp_target] 

            for key, output in zip(METHODS, temps):
                 correct, missed, false_d = checkIfCaught(temp_target, output)
                 fdr[key, 'correct'] += correct
                 fdr[key, 'missed']  += missed
                 fdr[key, 'false']   += false_d

        print("------"+str(cellline)+"--Chro:"+str(chro)+"-------")
        for key in METHODS:
            print(key+":"+str(fdr[key, 'correct'])+","+str(fdr[key, 'missed'])+","+str(fdr[key, 'false']))
            precision, recall, f1 = getMetrics(fdr[key, 'correct'], fdr[key, 'missed'], fdr[key, 'false'])
            final_scores[cellline, chro, key, 'precision'] = precision
            final_scores[cellline, chro, key, 'recall']    = recall
            final_scores[cellline, chro, key, 'f1']        = f1


#plot final scores
fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlabel("Precision")
ax.set_ylabel("Recall")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0,1.1)
ax.set_ylim(0,1.1)
for chro, size in zip(CHROS,CHRO_SIZES):
    for cellline, shape in zip(CELL_LINES, CELL_LINE_SHAPES):
        for meth, color in zip(METHODS, METHOD_COLORS):
            ax.scatter(final_scores[cellline, chro, meth, 'precision'], final_scores[cellline, chro, meth, 'recall'], s=100, c=color, marker=shape)
plt.show()
            
pdb.set_trace()
