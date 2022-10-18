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
    csv_file = pd.read_csv('CN4_rotations/overlap.csv')
    
    
    
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
        total_datapoints=i
        
    confidence_array = np.zeros(M)
    idx=0
    for b in binned_data:
        confidence_array[b-1]+=softmax_probs[idx]
        idx+=1
    accuracy = bin_array[:,2]
    i=0
    ECE=0
    for arr in bin_array:
        accuracy[i] = bin_array[i,0]/bin_array[i,1]
        confidence_array[i] /= bin_array[i,1]
        ECE+=(bin_array[i,1]/total_datapoints)*abs(bin_array[i,2]-confidence_array[i])
        i+=1
    ECE = np.format_float_positional(ECE, precision=4)
    print(type(ECE))
    
    bins = bins + 1/(2*M) #shift each bin so that it is central
    plt.subplot(111)
    plt.plot(bins[:M], bin_array[:M, 2])
    
    x = np.linspace(0,1,10)
    y =np.linspace(0,1,10)
    plt.plot(x,y)
    plt.grid
    plt.title("Model Calibration Plot: Lenet Test")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    textstr = ("ECE =  " + ECE)
    props = dict(boxstyle='square', facecolor='silver', alpha=0.5)
    plt.text(0.05, 0.95, textstr, fontsize=14,
        verticalalignment='top', bbox=props)
    plt.show()
    plt.savefig("calibration_plot_CN4_test")
    
    return

main()

 