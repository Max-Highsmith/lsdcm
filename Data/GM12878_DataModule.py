import matplotlib.pyplot as plt
import os
import sys
from utils import utils as ut
import pdb
import subprocess
import glob
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import random_split, DataLoader, Dataset


class GM12878Module(pl.LightningDataModule):
    def __init__(self,
            batch_size   = 64,
            res          = 10000,
            juicer_tool  = "other_tools/juicer_tools_1.22.01.jar"):

        self.juicer_tool = juicer_tool
        self.batch_size  = batch_size
        self.res         = res
        self.low_res_fn  = "Data/GSM1551550_HIC001_30.hic"
        self.hi_res_fn   = "Data/GSE63525_GM12878_insitu_primary_30.hic"
        self.piece_size  = 300
        self.step        = 50

    def download_raw_data(self):
        globs = glob.glob("Data/GSM1551550_HIC001_30.hic")
        found_data = (globs[0] == "Data/GSM1551550_HIC001_30.hic")
        if not found_data:
            print("downloading from GSE ... this could take a while")
            subprocess.run("bash scripts/getSmallData.sh", shell=True)     
        else:
            print("data found")

        globs = glob.glob("Data/GSE63525_GM12878_insitu_primary_30.hic")
        found_data = (globs[0] == "Data/GSE63525_GM12878_insitu_primary_30.hic")
        if not found_data:
            print("downloading from GSE ... this could take a while")
            subprocess.run("bash scripts/getSmallData.sh", shell=True)     #TODO fix
        else:
            print("data found")




    def extract_constraint_mats(self):
        #extract hi res
        if not os.path.exists("Data/Constraints"):
            subprocess.run("mkdir Data/Constraints", shell=True)
        for i in range(1,23):
            juice_command = "java -jar "\
                   ""+str(self.juicer_tool)+" dump observed KR "\
                   ""+str(self.hi_res_fn)+" "+str(i)+" "+str(i)+""\
                   " BP "+str(self.res)+" Data/Constraints/high_chr"+str(i)+"_res_"+str(self.res)+".txt"
            subprocess.run(juice_command, shell=True)
            juice_command = "java -jar "\
                   ""+str(self.juicer_tool)+" dump observed KR "\
                   ""+str(self.low_res_fn)+" "+str(i)+" "+str(i)+""\
                   " BP "+str(self.res)+" Data/Constraints/low_chr"+str(i)+"_res_"+str(self.res)+".txt"
            subprocess.run(juice_command, shell=True)

    def extract_create_numpy(self):
        if not os.path.exists("Data/Full_Mats"):
            subprocess.run("mkdir Data/Full_Mats", shell=True)

        for i in range(6,23):
           target, data = ut.loadBothConstraints("Data/Constraints/high_chr"+str(i)+"_res_"+str(self.res)+".txt",
                               "Data/Constraints/low_chr"+str(i)+"_res_"+str(self.res)+".txt",
                                self.res)       
           np.save("Data/Full_Mats/gm12878_mat_high_chr"+str(i)+"_res_"+str(self.res), target)
           np.save("Data/Full_Mats/gm12878_mat_low_chr"+str(i)+"_res_"+str(self.res), data)

    def split_numpy(self):
        if not os.path.exists("Data/Splits"):
            subprocess.run("mkdir Data/Splits", shell=True)
        for i in range(1,23):
            target =  ut.splitPieces("Data/Full_Mats/gm12878_mat_high_chr"+str(i)+"_res_"+str(self.res)+".npy",self.piece_size, self.step)
            data   =  ut.splitPieces("Data/Full_Mats/gm12878_mat_low_chr"+str(i)+"_res_"+str(self.res)+".npy", self.piece_size, self.step)
            np.save("Data/Splits/gm12878_high_chr_"+str(i)+"_res_"+str(self.res), target)
            np.save("Data/Splits/gm12878_low_"+str(i)+"_res_"+str(self.res), data)
        
    def prepare_data(self):
        print("Prepare the Preparations")
        #globs      = glob.glob("Data/GSM1551550_HIC001_30.hic")
        #found_data = (globs[0] == "Data/GSM1551550_HIC001_30.hic")

    
    class gm12878Dataset(Dataset):
            def __init__(self, tvt, res):
                self.tvt = tvt
                self.res = res
                if tvt   == "train":
                    self.chros = [1,3,5,6,7,9,11,12,13,15,17,18,19,21]
                elif tvt == "val":
                    self.chros = [2,8,10,22]
                elif tvt == "test":
                    self.chros = [4,14,16,20]

                self.data = np.load("Data/Splits/gm12878_high_chr_"+str(self.chros[0])+"_res_"+str(self.res)+".npy")
                for c, chro in enumerate(self.chros[1:]):
                    temp = np.load("Data/Splits/gm12878_high_chr_"+str(chro)+"_res_"+str(self.res)+".npy")
                    self.data = np.concatenate((self.data, temp))

            def __len__(self):
                return self.data.shape[0]

            def __getitem__(self, idx):
                return self.data[idx]

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_set = self.gm12878Dataset('train', self.res)
            self.val_set   = self.gm12878Dataset('val', self.res)
        if stage == 'test':
            self.test_set  = self.gm12878Dataset('test', self.res)
    
    def train_dataloader(self):
            return DataLoader(self.train_set, self.batch_size)
    
    def val_dataloader(self):
            return DataLoader(self.val_set, self.batch_size)

    def test_dataloader(self):
            return DataLoader(self.test_set, self.batch_size)
