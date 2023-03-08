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
marker_size = 50
use_true_labels = 1
use_softmax = 0
use_ood = 1

def main():
    
    #plot_latent_space('gplenet_rotations/gplenet_latent_values_no_mom.csv', 'average overlap')
    #plot_latent_space('latent_values.csv', 'average overlap')
    plot_latent_space('latent_values.csv', 'predictive entropy')
    
    return

def plot_latent_space(filename, uncertainty_metric):
    
    
    plt.figure(figsize=(6.5,5))
    ax = plt.subplot(111)
    csvfile = pd.read_csv(filename)
    accuracy = 0
    inc_class = 0
    
    targets = csvfile['target'].values
    error_rate = csvfile['classification error %'].values
    p1 = csvfile['latent_parameter_1'].values
    p2 = csvfile['latent_parameter_2'].values
    uncertainty = np.sqrt(csvfile[uncertainty_metric].values)
    softmax_probs_str = csvfile['softmax prob'].values
    softmax_probs = np.empty([len(softmax_probs_str)])
    
    fr1 = []
    fr2 = []
    ood = []
    
    j=0
    for str in softmax_probs_str:
        temp = str.strip('[')
        softmax_probs[j] = (1 - float(temp.split()[0]))
        j += 1
        
    if use_true_labels:
        
        if use_softmax:
            #uncertainty = softmax_probs
            i = 0
            for j in targets:
                if targets[i] == 0:
                    uncertainty[i] = softmax_probs[i]
                if targets[i] == 1:
                    uncertainty[i] = 1-softmax_probs[i]
                
                i += 1
        
        i = 0
        for j in targets:
            if targets[i] == 0:
                if softmax_probs[i] > 0.5:
                    inc_class += 1
                    print(i)
                fr1.append([p1[i], p2[i], uncertainty[i]])
                
            if targets[i] == 1:
                if softmax_probs[i] < 0.5:
                    inc_class += 1
                    print(i)
                fr2.append([p1[i], p2[i], uncertainty[i]])
                
            if targets[i] == 2:
                ood.append([p1[i], p2[i], 1])
                
            i += 1
    
    test_samples=0
    for j in targets:
        if j == 0 or j == 1:
            test_samples += 1
    
    x1 = [x[0] for x in fr1]
    y1 = [x[1] for x in fr1]
    u1 = [x[2] for x in fr1]
    x2 = [x[0] for x in fr2]
    y2 = [x[1] for x in fr2]
    u2 = [x[2] for x in fr2]
    if use_ood:
        x3 = [x[0] for x in ood]
        y3 = [x[1] for x in ood]
        u3 = [x[2] for x in ood]
        
    accuracy = (test_samples-inc_class)*100/test_samples
    print('Model accuracy =', accuracy)
    
    
    #plt.grid(visible=True)
    #plt.tight_layout()
    
    #plt.xlim(0,1)
    #plt.ylim(0,1)
    # ax.spines.right.set_visible(False)
    # ax.spines.top.set_visible(False)
    # ax.spines.left.set_visible(False)
    # ax.spines.bottom.set_visible(False)
    
    plt.title('Latent space visualisation for ' + uncertainty_metric, fontsize=12)
    
    
    h0 = plt.scatter(x1, y1, c=u1, cmap='Blues', label='FR I', marker='.', s=marker_size)
    cbar1 = plt.colorbar(pad=-0.09)
    cbar1.set_label('Uncertainty')
    h1 = plt.scatter(x2, y2, c=u2, cmap='Oranges', label='FR II', marker='.', s=marker_size)  
    cbar2 = plt.colorbar()
    cbar2.set_ticks([])
    if use_ood:
        h3 = plt.scatter(x3, y3, c='green', label='OoD', marker='.', s=marker_size)  
    
    
    # ax = plt.gca()
    # leg = ax.get_legend()
    # leg.legendHandles[0].set_color('blue')
    # leg.legendHandles[1].set_color('orange')
    plt.legend()
    
    plt.show()
    
    return




main()