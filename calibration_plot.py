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
    csv_file = pd.read_csv(model_to_plot + '_rotations/' + model_to_plot + '_overlap_with_err.csv')
    equal_count_bins = True
    
#------------------------------------------------------------------------------
#Define variables

    bins = np.zeros(M)
    bin_centres = np.zeros(M)
    accuracy = np.zeros(M)
    confidence = np.zeros(M)
    uncertainty = np.zeros(M)
    error = np.zeros(M)
    error = np.zeros(M)
    ECE=0
    UCE=0
    
    error_bins = np.zeros(M)
    error_bin_centres = np.zeros(M)
    
#Extract data from csv file

    target = csv_file["target"].values
    softmax_probs_str = csv_file['softmax prob'].values
    softmax_probs = np.empty([len(softmax_probs_str)])
    overlaps = csv_file['average overlap'].values
    image_errors = csv_file['classification error %'].values
    total_datapoints = len(target)
    bin_widths_probs = np.zeros(M)
    bin_widths_uncert = np.zeros(M)
    # if(dataset_size % M != 0):
    #     bin_width += 1

    i = 0
    for str in softmax_probs_str:
        temp = str.strip('[')
        softmax_probs[i] = (1 - float(temp.split()[0]))
        i += 1
    
    #error_target = target
    softmax_probs_sorted = softmax_probs[softmax_probs.argsort()]
    #accuracy_sorted = 1 - errors[softmax_probs.argsort()]
    #print(accuracy_sorted)
    accuracy_target = target[softmax_probs.argsort()]
    overlap_sorted = overlaps[overlaps.argsort()]
    image_errors_sorted = image_errors[overlaps.argsort()]
    
    #error_target = error_target[overlaps.argsort()]
    #print(softmax_probs_sorted)
    #print(overlap_sorted)
    
#Bin data

    error_classifications = np.zeros(M)
    classifications = np.zeros(M)
    
#------------------------------------------------------------------------------
    #Equal count bins
    
    if(equal_count_bins): 
        bin_count = int(total_datapoints/M)
        i=0
        for i in range(M-1):
            bins[i] = np.quantile(softmax_probs_sorted, (i+1)/M)
            error_bins[i] = np.quantile(overlap_sorted, (i+1)/M)
            prev_value = 0
            prev_value_u = 0
            if i != 0:
                prev_value = bins[i-1]
                prev_value_u = error_bins[i-1]
            bin_centres[i] = (bins[i] - prev_value)/2 + prev_value
            error_bin_centres[i] = (error_bins[i] - prev_value_u)/2 + prev_value_u
        error_bins[M-1] = 1
        bins[M-1] = 1
        error_bin_centres[M-1] = (1 - error_bins[M-2])/2 + error_bins[M-2] 
        bin_centres[M-1] = (1 - bins[M-2])/2 + bins[M-2] 
        
        i = 0
        for b in range(M):
            for j in range(bin_count):
                confidence[b] += softmax_probs[j + (b)*bin_count] #running total of confidence
                uncertainty[b] += overlaps[j + (b)*bin_count] #running total of uncertainty
                if accuracy_target[j + (b)*bin_count] == 1:
                    classifications[b] += 1 #total classified in bin
                # if error_target[j + (b)*bin_width] == 1:
                #      error_classifications[b] += 1 #total classified in bin
                
            for j in range(bin_count):
                error_classifications[b] += image_errors_sorted[j + b*bin_count]
                
            accuracy[i] = classifications[i]/bin_count
            uncertainty[i] /= bin_count
            confidence[i] /= bin_count
            error[i] = error_classifications[i]/bin_count
            ECE += (bin_count/total_datapoints)*abs(accuracy[i] - confidence[i])
            UCE += (bin_count/total_datapoints)*abs(error[i] - uncertainty[i])
            
            #Use average value of the bin as the centre
            #bin_centres[i] = confidence[i]
            #error_bin_centres[i] = uncertainty[i]
            
            i += 1
#------------------------------------------------------------------------------
    else:
        bins = np.linspace(0, 0.4, M)
        binned_probs = np.digitize(softmax_probs, bins)
        binned_uncert = np.digitize(overlaps, bins)
        bin_count = np.zeros(M)
        bin_count_uncert = np.zeros(M)
        print(binned_probs)
        
        i = 0
        for b in binned_probs:
            confidence[b-1] += softmax_probs[i] #running total of confidence
            bin_count[b-1] += 1 #total in bin
            if target[i] == 1:
                classifications[b-1] += 1 #total classified in bin
            i += 1

        i = 0
        for b in binned_uncert:
            uncertainty[b-1] += overlaps[i] #running total of uncertainty
            bin_count_uncert[b-1] += 1 #total in bin
            error[b-1] += image_errors[i]
            i += 1
    
    #Calculate average accuracy,uncertainty,confidence and ECE and UCE
        
        print(uncertainty)
        print(bin_count_uncert)
        for i in range(M):
            accuracy[i] = classifications[i]/bin_count[i]
            uncertainty[i] /= bin_count_uncert[i]
            confidence[i] /= bin_count[i]
            error[i] /= bin_count_uncert[i]
            ECE += (bin_count[i]/total_datapoints)*abs(accuracy[i] - confidence[i])
            UCE += (bin_count_uncert[i]/total_datapoints)*abs(error[i] - uncertainty[i])
            
            #Use average value of the bin as the centre
            #bin_centres[i] = confidence[i]
            #error_bin_centres[i] = uncertainty[i]
            
#------------------------------------------------------------------------------
        
    

    
#Calculate the total confidence, uncertainty and number of points in each bin

    

#Format ECE and UCE, then plot graphs

    ECE = np.format_float_positional(ECE, precision=4)
    UCE = np.format_float_positional(UCE, precision=4)
    
    bins = bins + 1/(2*M) #shift each bin so that it is central
    plt.subplot(111)
    if(equal_count_bins):
        plt.scatter(bin_centres[:M], accuracy[:M], marker='x')
    else:
        plt.scatter(bins[:M], accuracy[:M], marker='x')
    
    x = np.linspace(0, 1.0, 10)
    y = np.linspace(0, 1.0, 10)
    plt.plot(x,y, '--', linewidth=1)
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
    if(equal_count_bins):
        plt.scatter(error_bin_centres[:M], error[:M], marker='x')
    else:
        plt.scatter(bins[:M], error[:M], marker='x')
    
    
    x = np.linspace(0,1,10)
    y = np.linspace(0,1,10)
    plt.plot(x,y, '--',linewidth=1)
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

 