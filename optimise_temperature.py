# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 12:11:52 2023

@author: samhu
"""

import torch
import pandas as pd

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchsummary import summary

import numpy as np
import scipy as sc
import csv
import matplotlib.pyplot as plt
from PIL import Image

from models import VanillaLeNet, CNSteerableLeNet, DNSteerableLeNet, DNRestrictedLeNet
from utils import *
from FRDEEP import FRDEEPF
from MiraBest import MBFRConfident
#from MingoLoTSS import MLFR

from plots import plot_image

#------------------------------------------------------------------------------

def softmax(logits, temperature):
    
    return (np.exp(logits[1]/temperature)/(np.exp(logits[0]/temperature) + np.exp(logits[1]/temperature)))

def calculate_ECE(logit0, logit1, temperature, image_errors, target):
#Calculate ECE for a given temperature
    
    M = 26 #number of bins for histogram
    
    #Get softmax values from scaled logits
    softmax_probs = np.zeros(len(target))
    for i in range(0, len(target)):
        softmax_probs[i] = softmax([logit0[i], logit1[i]], temperature)
        
    #Define variables and arrays
    bins = np.zeros(M)
    bin_centres = np.zeros(M)
    accuracy = np.zeros(M)
    confidence = np.zeros(M)
    ECE=0
    softmax_probs_sorted = softmax_probs[softmax_probs.argsort()]
    accuracy_sorted = (1 - image_errors[softmax_probs.argsort()])
    accuracy_target = target[softmax_probs.argsort()]
    classifications = np.zeros(M)
    total_datapoints = len(target) 
    
    bin_count = int(total_datapoints/M)
    i=0
    for i in range(M-1): #create the bins and their min/max values
        bins[i] = np.quantile(softmax_probs_sorted, (i+1)/M)
        prev_value = 0
        
        if i != 0:
            prev_value = bins[i-1]
            
        bin_centres[i] = (bins[i] - prev_value)/2 + prev_value
        
    bins[M-1] = 1
    bin_centres[M-1] = (1 - bins[M-2])/2 + bins[M-2]
    
    i = 0
    for b in range(M): # actually bin the data
        
        for j in range(0, bin_count):
            
            confidence[b] += softmax_probs_sorted[j + (b)*bin_count] #running total of confidence
        
        for j in range(bin_count):
            if accuracy_target[j + b*bin_count] == 1:
                classifications[b] += accuracy_sorted[j + b*bin_count]
                
            else:
                classifications[b] += 1 - accuracy_sorted[j + b*bin_count]
                
        accuracy[i] = classifications[i]/bin_count #average over bin
        confidence[i] /= bin_count
        
        ECE += (bin_count/total_datapoints)*abs(accuracy[i] - confidence[i])
        
        #Use average value of the bin as the centre
        bin_centres[b] = confidence[b]

        i += 1
        
    return ECE


def main():
    
    model_to_plot = 'C4'
    csv_file = pd.read_csv(model_to_plot + '_rotations/' + model_to_plot + '_latent_values_beta=1.csv')
    number_of_points = 100
    
#Extract data from csv file
    target = csv_file["target"].values
    image_errors = csv_file['classification error %'].values
    logit0 = csv_file['latent_parameter_1'].values
    logit1 = csv_file['latent_parameter_2'].values
    
    temperature_values = np.linspace(0.1, 1, number_of_points)
    ECE_values = np.zeros(number_of_points)
    
    i=0
    min_ECE = 1
    opt_temp = 1
    for t in temperature_values:
        ECE_values[i] = calculate_ECE(logit0, logit1, t, image_errors, target)
        if ECE_values[i] < min_ECE:
            min_ECE = ECE_values[i]
            opt_temp = temperature_values[i]
        i+=1
    
    #print(temperature_values)
    #print(ECE_values)
    ECE = np.format_float_positional(min_ECE, precision=4)
    T = np.format_float_positional(opt_temp, precision=2)
    
    plt.plot(temperature_values, ECE_values)
    plt.title('Calibration score as function of Temperature: ' + model_to_plot)
    plt.grid(visible=True)
    plt.xlim(0,1)
    plt.xlabel('Temperature')
    plt.ylabel('ECE')
    textstr = ("Best ECE =  " + ECE + "\n" "Optimal Temp = " + T)
    props = dict(boxstyle='square', facecolor='silver', alpha=0.5)
    plt.text(0.05, 0.047, textstr, fontsize=14, bbox=props)
    plt.show()
    
    print('The best ECE is ', min_ECE, ' at a temperature of ', opt_temp)
    
    return
    
main()