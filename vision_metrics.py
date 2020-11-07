import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import argparse
import sys
sys.path.append(".")
sys.path.append("../")
import numpy as np
from sklearn.decomposition import PCA
import glob
import yaml
import matplotlib.pyplot as plt
import torch
import pdb
from pytorch_lightning import Trainer
from Data.GM12878_DataModule import GM12878Module
from GAN_Module import GAN_Model

metric_logs = {
        "pre_pcc":[],
        "pas_pcc":[],
        "pre_spc":[],
        "pas_spc":[],
        "pre_psnr":[],
        "pas_psnr":[],
        "pre_ssim":[],
        "pas_ssim":[],
        "pre_mse":[],
        "pas_mse":[],
        "pre_snr":[],
        "pas_snr":[]
        }

def _toimg(mat):
    m = torch.tensor(mat)
    # convert to float and add channel dimension
    return m.float().unsqueeze(0)

def _tohic(mat):
    mat.squeeze_()
    return mat.numpy()#.astype(int)

def gaussian(width, sigma):
    gauss = torch.Tensor([exp(-(x-width//2)**2 / float(2 * sigma**2)) for x in range(width)])
    return gauss / gauss.sum()

def create_window(window_size, channel, sigma=3):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def gaussian_filter(img, width, sigma=3):
    img = _toimg(img).unsqueeze(0)
    _, channel, _, _ = img.size()
    window = create_window(width, channel, sigma)
    mu1 = F.conv2d(img, window, padding=width // 2, groups=channel)
    return _tohic(mu1)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    img1 = _toimg(img1).unsqueeze(0)
    img2 = _toimg(img2).unsqueeze(0)
    _, channel, _, _ = img1.size()
    window = create_window(window_size, channel)
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def logSSIM(data, target, output):
    metric_logs['pre_ssim'].append(compareSSIM(data, target))
    metric_logs['pas_ssim'].append(compareSSIM(output, target))

def logPSNR(data, target, output):
    metric_logs['pre_psnr'].append(comparePSNR(data, target))
    metric_logs['pas_psnr'].append(comparePSNR(output, target))

def logPCC(data, target, output):
    metric_logs['pre_pcc'].append(comparePCC(data, target))
    metric_logs['pas_pcc'].append(comparePCC(output, target))

def logSPC(data, target, output):
    metric_logs['pre_spc'].append(compareSPC(data, target))
    metric_logs['pas_spc'].append(compareSPC(output, target))

def logMSE(data, target, output):
    metric_logs['pre_mse'].append(compareMSE(data, target))
    metric_logs['pas_mse'].append(compareMSE(output, target))

def logSNR(data, target, output):
    metric_logs['pre_snr'].append(compareSNR(data, target))
    metric_logs['pas_snr'].append(compareSNR(output, target))

def compareSPC(a, b):
    return spearmanr(a[0][0], b[0][0], axis=None)[0]

def comparePCC(a, b):
    return pearsonr(a[0][0].flatten(), b[0][0].flatten())[0]

def comparePSNR(a, b):
    MSE = np.square(a[0][0]-b[0][0]).mean().item()
    MAX = torch.max(b).item()
    return 20*np.log10(MAX) - 10*np.log10(MSE)

def compareSNR(a, b):
    return torch.sum(b[0][0]).item()/torch.sqrt(torch.sum((b[0][0]-a[0][0])**2)).item()

def compareSSIM(a, b):
    return ssim(a, b).item()

def compareMSE(a, b):
    return np.square(a[0][0]-b[0][0]).mean().item()

def log_means(name):
    return (name, np.mean(metric_logs[name]))


ssim = SSIM()

parser = argparse.ArgumentParser()
parser.add_argument("version")
args  =  parser.parse_args()

VERSION = args.version
PATH    = glob.glob("lightning_logs/version_"+str(VERSION)+"*/checkpoints/*")[0]
op = open("lightning_logs/version_"+str(VERSION)+"/hparams.yaml")
hparams = yaml.load(op)
print(hparams)
dm_test      = GM12878Module(batch_size=1)
dm_test.prepare_data()
dm_test.setup(stage='test')

model   = GAN_Model()

pretrained_model = model.load_from_checkpoint(PATH)
pretrained_model.freeze()

for e, epoch in enumerate(dm_test.test_dataloader()):
    print(str(e)+"/"+str(dm_test.test_dataloader().dataset.data.shape[0]))
    data, full_target, info = epoch
    target      = full_target[:,:,6:-6,6:-6]
    filter_data = data[:,:,6:-6,6:-6]
    output             = pretrained_model(data)
    logPCC(data=filter_data, target=target, output=output)
    logSPC(data=filter_data, target=target, output=output)
    logMSE(data=filter_data, target=target, output=output)
    logPSNR(data=filter_data, target=target, output=output)
    logSNR(data=filter_data, target=target, output=output)
    logSSIM(data=filter_data, target=target, output=output)
print(list(map(log_means, metric_logs.keys())))


