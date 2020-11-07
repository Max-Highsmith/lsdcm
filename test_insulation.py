import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pdb
import torch
from loss.insulation import computeInsulation
from Data.GM12878_DataModule import GM12878Module

dm = GM12878Module()
dm.setup(stage='fit')
ds = torch.tensor(dm.train_dataloader().dataset.target[0:35])
ci = computeInsulation()
iv, dv, boundaries = ci(ds)
for i in range(0,15):
    fig, ax = plt.subplots()
    ax.imshow(ds[i][0], cmap="Oranges")
    for bound in boundaries[i]:
        ax.axvline(x=bound)
        rect = patches.Rectangle((bound-25,bound-25),50,50, facecolor='none', edgecolor='r')
        ax.add_patch(rect)
    ax.plot(list(range(24, len(iv[i][0])+24)),iv[i][0]*100+50, label="iv")
    ax.plot(list(range(35, len(dv[i][0])+35)),dv[i][0]*100+200, label="dv")
    plt.legend()
    plt.show()
pdb.set_trace()
