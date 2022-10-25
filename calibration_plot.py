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
#Parameters

    model_to_plot = 'CN8'
    M = 13 #number of bins for histogram
    csv_file = pd.read_csv(model_to_plot + '_rotations/overlap.csv')
    
#------------------------------------------------------------------------------
#Define variables

    bins = np.zeros(M)
    bin_centres = np.zeros(M)
    accuracy = np.zeros(M)
    confidence = np.zeros(M)
    uncertainty = np.zeros(M)
    error = np.zeros(M)
    ECE=0
    UCE=0
    
#Extract data from csv file

    target = csv_file["target"].values
    softmax_probs_str = csv_file['softmax prob'].values
    softmax_probs = np.empty([len(softmax_probs_str)])
    overlaps = csv_file['average overlap'].values
    dataset_size = len(target)
    bin_width = int(dataset_size / M)
    # if(dataset_size % M != 0):
    #     bin_width += 1
    
    #print(bin_width)
    #print(overlaps)
    
    i = 0
    for str in softmax_probs_str:
        temp = str.strip('[')
        softmax_probs[i] = (1 - float(temp.split()[0]))
        i += 1
    
    softmax_probs_sorted = softmax_probs[softmax_probs.argsort()]
    target = target[softmax_probs.argsort()]
    overlaps = overlaps[softmax_probs.argsort()]
    
    #print(softmax_probs_sorted)
    #print(overlaps)
    
#Bin data
    
    i=0
    for i in range(M-1):
        bins[i] = np.quantile(softmax_probs_sorted, (i+1)/M)
        prev_value = 0
        if i != 0:
            prev_value = bins[i-1]
        bin_centres[i] = (bins[i] - prev_value)/2 + prev_value
    bins[M-1] = 1
    bin_centres[M-1] = (1 - bins[M-2])/2 + bins[M-2] 
    
    print(bin_centres)
    
    classifications = np.zeros(M)
    bin_sizes = np.zeros(M)
    
#Calculate the total confidence, uncertainty and number of points in each bin

    i=0
    for b in range(M):
        for j in range(bin_width):
            print(j + (b)*bin_width)
            confidence[b-1] += softmax_probs[j + (b)*bin_width] #running total of confidence
            uncertainty[b-1] += overlaps[j + (b)*bin_width] #running total of uncertainty
            bin_sizes[b-1] += 1 #total in bin
            if target[j + (b-1)*bin_width] == 1:
                classifications[b-1] += 1 #total classified in bin
            i += 1
        
    total_datapoints = i
    
    print('The bin sizes are ', bin_sizes)

#Calculate average accuracy,uncertainty,confidence and ECE and UCE

    i=0
    for i in range(M):
        accuracy[i] = classifications[i]/bin_sizes[i]
        uncertainty[i] /= bin_sizes[i]
        confidence[i] /= bin_sizes[i]
        error[i] = (1 - accuracy[i])
        ECE += (bin_sizes[i]/total_datapoints)*abs(accuracy[i] - confidence[i])
        UCE += (bin_sizes[i]/total_datapoints)*abs(error[i] - uncertainty[i])
        i+=1
        
#Format ECE and UCE, then plot graphs

    ECE = np.format_float_positional(ECE, precision=4)
    UCE = np.format_float_positional(UCE, precision=4)
    
    bins = bins + 1/(2*M) #shift each bin so that it is central
    plt.subplot(111)
    plt.scatter(bin_centres[:M], accuracy[:M])
    
    x = np.linspace(0,1,10)
    y =np.linspace(0,1,10)
    plt.plot(x,y)
    plt.grid
    plt.title("Model Calibration Plot: " + model_to_plot + " Test data")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    textstr = ("ECE =  " + ECE)
    props = dict(boxstyle='square', facecolor='silver', alpha=0.5)
    plt.text(0.05, 0.95, textstr, fontsize=14,
        verticalalignment='top', bbox=props)
    plt.show()
    
    plt.subplot(111)
    plt.scatter(uncertainty[:M], error[:M])
    
    x = np.linspace(0,1,10)
    y =np.linspace(0,1,10)
    plt.plot(x,y)
    plt.grid
    plt.title("Model Calibration Plot: " + model_to_plot + " Test data")
    plt.xlabel("Uncertainty")
    plt.ylabel("Error")
    textstr = ("UCE =  " + UCE)
    props = dict(boxstyle='square', facecolor='silver', alpha=0.5)
    plt.text(0.05, 0.95, textstr, fontsize=14,
        verticalalignment='top', bbox=props)
    plt.show()
    
    return

main()

 