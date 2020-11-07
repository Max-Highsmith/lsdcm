import torch.nn.functional as F
import torch.nn as nn
import HiCSR_model as hc
import torch
import pdb
import pytorch_lightning as pl
from pytorch_lightning import Trainer

class GAN_Model(pl.LightningModule):
    def __init__(self):
        super(GAN_Model, self).__init__()
        self.generator     = hc.Generator(num_res_blocks=5)
        self.discriminator = hc.Discriminator() 
        self.G_lr = 0.001
        self.D_lr = 0.001
        self.generator.init_params()
        self.discriminator.init_params()
        
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.L1Loss()
    
    def forward(self, x):
        fake = self.generator(x)
        return fake

    def adversarial_loss(self, target, output):
        return self.bce(target, output)

    def meanSquaredError_loss(self, target, output):
        return self.mse(target, output)

    def training_step(self, batch, batch_idx, optimizer_idx):
        data, full_target = batch
        target = full_target[:,:,6:-6,6:-6]
        #Generator
        if optimizer_idx == 0:
            output      = self.generator(data)
            pred_fake   = self.discriminator(output)
            labels_real = torch.ones_like(pred_fake, requires_grad=False)
            MSE_loss    = self.meanSquaredError_loss(output, target)
            GAN_loss    = self.adversarial_loss(pred_fake, labels_real)
            
            total_loss_G = MSE_loss+GAN_loss
            self.log("total_loss_G", total_loss_G)
            return total_loss_G
        
        #Discriminator
        if optimizer_idx == 1:
            self.discriminator.zero_grad()

            #train on real data
            pred_real       = self.discriminator(target)
            labels_real     = torch.ones_like(pred_real, requires_grad=False)
            pred_labels_real = (pred_real>0.5).float()
            acc_real        = (pred_labels_real == labels_real).float().sum()/labels_real.shape[0]
            loss_real       = self.adversarial_loss(pred_real, labels_real)
            
            #train on fake data
            output           = self.generator(data)
            pred_fake        = self.discriminator(output.detach())
            labels_fake      = torch.zeros_like(pred_fake, requires_grad=False)
            pred_labels_fake = (pred_fake > 0.5).float()
            acc_fake         = (pred_labels_fake == labels_fake).float().sum()/labels_fake.shape[0]
            loss_fake        = self.adversarial_loss(pred_fake, labels_fake)

            total_loss_D = loss_real + loss_fake
            self.log("total_loss_D",total_loss_D)
            return total_loss_D



    def validation_step(self, batch, batch_idx):
        print("todo")

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.G_lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.D_lr)
        return [opt_g, opt_d]
