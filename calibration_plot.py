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
    csv_file = pd.read_csv('CN4_rotations/overlap_NO_ROT.csv')
    
    
#------------------------------------------------------------------------------
    
    bins = np.linspace(0,1,M+1,endpoint=True)
    accuracy = np.zeros(M)
    confidence = np.zeros(M)
    uncertainty = np.zeros(M)
    ECE=0
    UCE=0
    
#Extract data from csv file

    target = csv_file["target"].values
    softmax_probs_str = csv_file['softmax prob'].values
    softmax_probs = np.empty([len(softmax_probs_str)])
    overlaps = csv_file['average overlap'].values
    
    i = 0
    for str in softmax_probs_str:
        temp = str.strip('[')
        softmax_probs[i] = (1 - float(temp.split()[0]))
        i += 1
    
#Bin data
    
    binned_data = np.digitize(softmax_probs, bins)
    
    classifications = np.zeros(M)
    bin_sizes = np.zeros(M)
    
    i=0
    for b in binned_data:
        confidence[b-1] += softmax_probs[i] #running total of confidence
        uncertainty[b-1] += overlaps[i] #running total of uncertainty
        bin_sizes[b-1] += 1 #total in bin
        if target[i] == 1:
            classifications[b-1] += 1 #total classified in bin
        i += 1
    total_datapoints = i

#Calculate average accuracy,uncertainty,confidence and ECE and UCE

    i=0
    for i in range(M):
        accuracy[i] = classifications[i]/bin_sizes[i]
        uncertainty[i] /= bin_sizes[i]
        confidence[i] /= bin_sizes[i]
        ECE += (bin_sizes[i]/total_datapoints)*abs(accuracy[i] - confidence[i])
        i+=1
    print(uncertainty)
        
#Format ECE and UCE, then plot graphs

    ECE = np.format_float_positional(ECE, precision=4)
    
    bins = bins + 1/(2*M) #shift each bin so that it is central
    plt.subplot(111)
    plt.plot(bins[:M], accuracy[:M])
    
    x = np.linspace(0,1,10)
    y =np.linspace(0,1,10)
    plt.plot(x,y)
    plt.grid
    plt.title("Model Calibration Plot: CN4 Test data")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    textstr = ("ECE =  " + ECE)
    props = dict(boxstyle='square', facecolor='silver', alpha=0.5)
    plt.text(0.05, 0.95, textstr, fontsize=14,
        verticalalignment='top', bbox=props)
    plt.show()
    
    return

main()

 