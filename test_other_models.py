from Data.GM12878_DataModule import GM12878Module
import matplotlib.pyplot as plt
import pdb
import torch
import numpy

#other models
import models_to_compare.models.deephic as deephic
import models_to_compare.models.hicplus as hicplus
import models_to_compare.models.hicsr as hicsr



#load data
dm_test = GM12878Module(batch_size=1, piece_size=269)
dm_test.prepare_data()
dm_test.setup(stage='test')
ds     = torch.from_numpy(dm_test.test_dataloader().dataset.data[0:1])
target = torch.from_numpy(dm_test.test_dataloader().dataset.target[0:1])

#TODO  there is no hicGAN because there isn't a pytorch version available
#TODO  hiCNN doesn't have pretrained weights
#load models

model_hicplus = hicplus.Net(40,28)
model_hicplus.load_state_dict(torch.load("models_to_compare/weights/pytorch_HindIII_model_40000"))

model_deephic = deephic.Generator(scale_factor=1, in_channel=1, resblock_num=5)
model_deephic.load_state_dict(torch.load("models_to_compare/weights/deephic_kr_16.pth"))

model_hicsr   = hicsr.Generator(num_res_blocks=15)
model_hicsr.load_state_dict(torch.load("models_to_compare/weights/HiCSR.pth"))

#pass through models
FULL_RES    = 269
hicplus_out = torch.zeros((269,269))
for i in range(0, FULL_RES-40, 28):
    for j in range(0, FULL_RES-40, 28):
        temp                  = ds[:,:,i:i+40, j:j+40]
        print("i:",str(i)," j:",str(j), temp.shape)
        hicplus_out[i+6:i+34, j+6:j+34] =  model_hicplus(temp)#temp[0][0][6:34,6:34] #model_hicplus(temp)
hicplus_out = hicplus_out.detach()

hicsr_out = torch.zeros((269,269))
for i in range(0, FULL_RES-40, 28):
    for j in range(0, FULL_RES-40, 28):
        temp                  = ds[:,:,i:i+40, j:j+40]
        print("i:",str(i)," j:",str(j), temp.shape)
        hicsr_out[i+6:i+34, j+6:j+34] =  model_hicsr(temp)

hicsr_out = hicsr_out.detach()

deephic_out = torch.zeros((FULL_RES, FULL_RES))
for i in range(0, FULL_RES-40, 28):
    for j in range(0, FULL_RES, 28):
        temp                        = ds[:,:,i:i+40, j:j+40]
        deephic_out[i:i+40, j:j+40] = model_deephic(temp)

deephic_out = deephic_out.detach()

lowres_out  = ds[0][0]
target_out  = target[0][0]


#show comparison plots
fig, ax = plt.subplots(1,5)
for i in range(0, 5):
    ax[i].set_xticks([])
    ax[i].set_yticks([])
ax[0].imshow(lowres_out, cmap="Reds")
ax[1].imshow(hicplus_out, cmap="Reds")
ax[2].imshow(deephic_out, cmap="Reds")
ax[3].imshow(hicsr_out, cmap="Reds")
ax[4].imshow(deepchromap_out, cmap="Reds")
ax[4].imshow(target_out, cmap="Reds")

ax[0].set_title("DownSampled")
ax[1].set_title("Hicplus")
ax[2].set_title("Deephic")
ax[3].set_title("Hicsr")
ax[4].set_title("DeepChroMap")
ax[5].set_title("target")
plt.show()

pdb.set_trace()




