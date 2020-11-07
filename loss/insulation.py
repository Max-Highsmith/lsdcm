import torch.nn.functional as F
import matplotlib.pyplot as plt
import pdb
import numpy as np
import torch

class computeInsulation(torch.nn.Module):
    def __init__(self):
        super(computeInsulation, self).__init__()
        self.di_pool     = torch.nn.AvgPool2d(kernel_size=51, stride=1)
        self.top_pool    = torch.nn.AvgPool1d(kernel_size=10, stride=1)
        self.bottom_pool = torch.nn.AvgPool1d(kernel_size=10, stride=1)
    
    def forward(self, x):
        iv     = self.di_pool(x)
        iv     = torch.diagonal(iv, dim1=2, dim2=3)       
        iv     = torch.log2(iv/torch.mean(iv))
        top    = self.top_pool(iv[:,:,10:])
        bottom = self.bottom_pool(iv[:,:,:-10])
        dv     = (top-bottom)
        left   = torch.cat([torch.zeros(dv.shape[0], dv.shape[1],2), dv], dim=2)
        right  = torch.cat([dv, torch.zeros(dv.shape[0], dv.shape[1],2)], dim=2)
        band   = ((left<0) == torch.ones_like(left)) * ((right>0) == torch.ones_like(right))
        band   = band[:,:,2:-2]
        boundaries = []
        for i in range(0, band.shape[0]):
            cur_bound = torch.where(band[i,0])[0]+36
            boundaries.append(cur_bound)
            print(cur_bound)
        return iv, dv, boundaries

class InsulationLoss(torch.nn.Module):
    def __init__(self):
        super(InsulationLoss, self).__init__()
        self.di_pool     = torch.nn.AvgPool2d(kernel_size=51, stride=1)
        self.top_pool    = torch.nn.AvgPool1d(kernel_size=10, stride=1)
        self.bottom_pool = torch.nn.AvgPool1d(kernel_size=10, stride=1)

    def indivInsulation(self, x):
        iv     = self.di_pool(x)
        iv     = torch.diagonal(iv, dim1=2, dim2=3)       
        iv     = torch.log2(iv/torch.mean(iv))
        top    = self.top_pool(iv[:,:,10:])
        bottom = self.bottom_pool(iv[:,:,:-10])
        dv     = (top-bottom)
        return dv

    def forward(self, output, target):
        out_dv = self.indivInsulation(output)
        tar_dv = self.indivInsulation(target)
        loss   = F.mse_loss(tar_dv, out_dv)
        return loss
        
