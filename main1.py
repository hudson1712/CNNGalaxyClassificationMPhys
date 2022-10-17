# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 14:34:02 2022
Samuel Hudson
"""

import torch
import pandas as pd

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchsummary import summary

import numpy as np
import csv
import matplotlib.pyplot as plt
from PIL import Image

from models import VanillaLeNet, CNSteerableLeNet, DNSteerableLeNet, DNRestrictedLeNet
from utils import *
from FRDEEP import FRDEEPF
from MiraBest import MBFRConfident
#from MingoLoTSS import MLFR

from plots import plot_image

def main():
    
    if(torch.cuda.is_available()):
        print('CUDA on')
    else:
        print("CUDA not available")

#------------------------------------------------------------------------------
#   Parameters

    M = 5 #number of bins for historgram
    csv_file = pd.read_csv('LeNet_rotations/overlap.csv')
    
    
    
#------------------------------------------------------------------------------
    
    bins = np.linspace(0,1,M+1,endpoint=True)
    
    print(bins)
    
#Extract data from csv file

    target  = csv_file["target"].values
    softmax_probs_str  = csv_file['softmax prob'].values
    softmax_probs = np.empty([len(softmax_probs_str)])
    i = 0
    for str in softmax_probs_str:
        temp = str.strip('[')
        softmax_probs[i] = (1 - float(temp.split()[0]))
        i += 1
    print(softmax_probs)
        
#Bin data
    
    binned_data = np.digitize(softmax_probs, bins)
        
    print(len(binned_data))
    print(binned_data)
    
    bin_array = np.zeros([M,3])
    i=0
    for b in binned_data:
        bin_array[b-1][1] += 1 #total in bin
        if target[i] == 1:
            bin_array[b-1][0] += 1 #total classified in bin
        i += 1
    
    i=0
    for arr in bin_array:
        bin_array[i,2] = bin_array[i,0]/bin_array[i,1]
        i+=1
        
    print(bin_array)
    
    bins = bins + 1/(2*M) #shift each bin so that it is central
    plt.subplot(111)
    plt.plot(bins[:M], bin_array[:M, 2])
    
    
    x = np.linspace(0,1,10)
    y =np.linspace(0,1,10)
    plt.plot(x,y)
    plt.grid
    plt.show()
    
    return

main()

 