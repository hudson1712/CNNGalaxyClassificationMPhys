# -*- coding: utf-8 -*-
import torch

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchsummary import summary
import pandas as pd
import os
from torchvision.io import read_image
from torchvision.io import ImageReadMode

import numpy as np
import csv
import glob
from PIL import Image
import matplotlib as plt
import umap
from sklearn.preprocessing import StandardScaler

from models import VanillaLeNet, CNSteerableLeNet, DNSteerableLeNet, GPLeNet, DN_GPLeNet
from utils import *
from plots import *
from FRDEEP import FRDEEPF
from MiraBest import *
from galaxy_mnist import *
import matplotlib.lines as mlines

def main():
    
    marker_size=25
    
    latent_features = np.genfromtxt('feature_space.csv', delimiter=',')
    output_uncerts = pd.read_csv('latent_values.csv')
    labels = np.genfromtxt('feature_labels.csv', delimiter=',', dtype=int)
    reducer = umap.UMAP(n_neighbors=50,
                        min_dist=0.1,
                        metric='correlation')
    scaled_features = StandardScaler().fit_transform(latent_features)
    embedding = reducer.fit_transform(scaled_features)
    embedding /= np.max(embedding)
    print(embedding.shape)
    
    uncertainties = np.sqrt(output_uncerts['variance'].values)
    uncertainties /= max(uncertainties)
    
    palette = ['Blue','Red','Green']
    
        
    fr1 = []
    fr2 = []
    ood = []
    
    i = 0
    for j in labels:
        if labels[i] == 0:
            fr1.append([embedding[i,0], embedding[i,1], uncertainties[i]])
            
        if labels[i] == 1:
            fr2.append([embedding[i,0], embedding[i,1], uncertainties[i]])
            
        if labels[i] == 2:
            ood.append([embedding[i,0], embedding[i,1], uncertainties[i]])
            
        i += 1
    
    x1 = [x[0] for x in fr1]
    y1 = [x[1] for x in fr1]
    u1 = [x[2] for x in fr1]
    x2 = [x[0] for x in fr2]
    y2 = [x[1] for x in fr2]
    u2 = [x[2] for x in fr2]
    x3 = [x[0] for x in ood]
    y3 = [x[1] for x in ood]
    u3 = [x[2] for x in ood]
    

    
    
    #objects = plt.scatter(embedding[:,0], embedding[:,1], c=[palette[labels[x].item()] for x in range(0,len(embedding[:,0]))])
    
    h0 = plt.scatter(x1, y1, c=u1, cmap='Blues', vmin=0, vmax=1, label='FR I', marker='.', s=marker_size)
    cbar1 = plt.colorbar(pad=-0.09)
    cbar1.set_label('Uncertainty')
    h1 = plt.scatter(x2, y2, c=u2, cmap='Oranges', vmin=0, vmax=1, label='FR II', marker='.', s=marker_size)  
    cbar2 = plt.colorbar()
    cbar2.set_ticks([])
    h3= plt.scatter(x3, y3, c=u3, cmap='Greens', vmin=0, vmax=1, label='OoD', marker='.', s=marker_size)  
    
    


    FR1 = mlines.Line2D([], [], color='blue', marker='.', ls='', label='FRI')
    FR2 = mlines.Line2D([], [], color='red', marker='.', ls='', label='FRII')
    OOD = mlines.Line2D([], [], color='green', marker='.', ls='', label='OOD')
    plt.legend(handles=[FR1, FR2, OOD])

    plt.title('UMAP of Feature Space')
    plt.show()
    return 0

main()