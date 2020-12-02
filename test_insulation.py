import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pdb
import torch
from loss.insulation import computeInsulation
from Data.GM12878_DataModule import GM12878Module
from Data.K562_DataModule    import K562Module
from Data.IMR90_DataModule   import IMR90Module
from Data.HMEC_DataModule    import HMECModule
import GAN_Module as deepchromap

#dm = GM12878Module()
WEIGHT_PATH       = "deepchromap_weights.ckpt"
deepChroModel     = deepchromap.GAN_Model()
model_deepchromap = deepChroModel.load_from_checkpoint(WEIGHT_PATH)

dm = K562Module()
dm.setup(stage=14)
low_ds = torch.tensor(dm.test_dataloader().dataset.data[0:35])
ds     = torch.tensor(dm.test_dataloader().dataset.data[0:35,:,6:-6,6:-6])
ts     = torch.tensor(dm.test_dataloader().dataset.target[0:35,:,6:-6,6:-6])
ci     = computeInsulation()
iv,   dv, boundaries  = ci(ds)
ivt, dvt, boundariest = ci(ts)
for i in range(15,35):
    print(i)
    fix, ax = plt.subplots(3,1, figsize=(8,10))
    vehicle_out = model_deepchromap(low_ds[i:i+1]).detach()
    ivv, dvv, boundariesv = ci(vehicle_out)
    ax[2].imshow(vehicle_out[0][0], cmap="Oranges")
    #ax[5].plot(range(ci.window_radius, len(ivv[0][0]) + ci.window_radius), ivv[0][0]*(100), color="red")
    #ax[5].plot(range(ci.window_radius+ci.deriv_size,
    #        len(dvv[0][0]) + ci.window_radius+ci.deriv_size),
    #        dvv[0][0]*(100), color="blue")
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    for bound in boundariesv[0]:
        ax[2].axvline(x=bound)

    ax[0].imshow(ds[i][0], cmap="Oranges")
    #ax[1].plot(range(ci.window_radius, len(iv[i][0]) + ci.window_radius), iv[i][0]*(100), color="red")
    #ax[1].plot(range(ci.window_radius+ ci.deriv_size,
    #        len(dv[i][0]) + ci.window_radius+ ci.deriv_size),
    #        dv[i][0]*(100), color="blue")
    for bound in boundaries[i]:
        ax[0].axvline(x = bound)

    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].imshow(ts[i][0], cmap="Oranges")
    #ax[3].plot(range(ci.window_radius, len(ivt[i][0]) + ci.window_radius), ivt[i][0]*(100), color="red")
    #ax[3].plot(range(ci.window_radius+ ci.deriv_size,
    #        len(dvt[i][0]) + ci.window_radius+ ci.deriv_size),
    #        dvt[i][0]*(100), color="blue")
    for bound in boundariest[i]:
        ax[1].axvline(x = bound)

    ax[1].set_xticks([])
    ax[1].set_yticks([])

    plt.show()
    '''
    fig, ax = plt.subplots()

    ax.plot(range(ci.window_radius+ ci.deriv_size,
            len(dvt[i][0]) + ci.window_radius+ ci.deriv_size),
            dvt[i][0]*(100), color="coral")
    ax.plot(range(ci.window_radius+ci.deriv_size,
            len(dvv[0][0]) + ci.window_radius+ci.deriv_size),
            dvv[0][0]*(100), color="yellow")
    ax.plot(range(ci.window_radius+ ci.deriv_size,
            len(dv[i][0]) + ci.window_radius+ ci.deriv_size),
            dv[i][0]*(100), color="black")
    plt.show()
    '''
pdb.set_trace()
