# -*- coding: utf-8 -*-
import pylab as pl
import pandas as pd
import numpy as np

import glob
from utils import *
from MiraBest import MBFRConfident

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import matplotlib.colors as colours

from matplotlib.ticker import ScalarFormatter,StrMethodFormatter

#pl.rcParams["font.family"] = "Times"
plt.rcParams['font.size'] = 12


def main():
    
    plot_latent_space('lenet_latent_test_data.csv', 'mutual information')
    
    return

def plot_latent_space(filename, uncertainty_metric):
    
    
    plt.figure(figsize=(6.5,5))
    ax = plt.subplot(111)
    csvfile = pd.read_csv(filename)
    
    targets = csvfile['target'].values
    p1 = csvfile['latent_parameter_1'].values
    p2 = csvfile['latent_parameter_2'].values
    uncertainty = csvfile[uncertainty_metric].values
    
    fr1 = []
    fr2 = []
    
    i = 0
    for j in targets:
        if targets[i] == 0:
            fr1.append([p1[i], p2[i], uncertainty[i]])
            
        if targets[i] == 1:
            fr2.append([p1[i], p2[i], uncertainty[i]])
            
        i += 1
        
    print(fr1)
    x1 = [x[0] for x in fr1]
    y1 = [x[1] for x in fr1]
    u1 = [x[2] for x in fr1]
    x2 = [x[0] for x in fr2]
    y2 = [x[1] for x in fr2]
    u2 = [x[2] for x in fr2]
    
    #plt.grid(visible=True)
    #plt.tight_layout()
    
    #plt.xlim(0,1)
    #plt.ylim(0,1)
    # ax.spines.right.set_visible(False)
    # ax.spines.top.set_visible(False)
    # ax.spines.left.set_visible(False)
    # ax.spines.bottom.set_visible(False)
    
    plt.title('Latent space visualisation for ' + uncertainty_metric, fontsize=12)
    
    
    plt.scatter(x1, y1, c=u1, cmap='Blues', norm=colours.LogNorm(vmin=0.00000001,vmax=1), label='FR I', marker='.', s=50)
    cbar1 = plt.colorbar(pad=-0.09)
    cbar1.set_label('Uncertainty')
    plt.scatter(x2, y2, c=u2, cmap='Oranges', norm=colours.LogNorm(vmin=0.00000001,vmax=1), label='FR II', marker='.', s=50)  
    cbar2 = plt.colorbar()
    cbar2.set_ticks([])
    plt.legend()
    
    plt.show()
    
    return




main()