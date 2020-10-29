import subprocess
import os
import matplotlib.pyplot as plt
import pdb
import sys
sys.path.append("Data")
sys.path.append(".")
sys.path.append("../")
import GM12878_DataModule as dm

mod = dm.GM12878Module()
mod.prepare_data()
#mod.split_numpy()
mod.setup('test')

if not os.path.exists("figs/gm12878_dataset"):
    if not os.path.exists("figs"):
        subprocess.run("mkdir figs", shell=True)
    subprocess.run("mkdir figs/gm12878_dataset", shell=True)

i = 0
for data_batch, target_batch in mod.test_dataloader():
    for sample in range(0, data_batch.shape[0]):
        print(str(i)+"/"+str(mod.test_dataloader().dataset.data.shape[0]))
        plt.imshow(target_batch[sample][0], cmap="Reds")
        plt.savefig("figs/gm12878_dataset/target"+str(i)+".png")

        plt.imshow(data_batch[sample][0], cmap="Reds")
        plt.savefig("figs/gm12878_dataset/data"+str(i)+".png")
        i = i+1


