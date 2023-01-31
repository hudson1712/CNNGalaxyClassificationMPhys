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

def main():
    
    if(torch.cuda.is_available()):
        print('CUDA on')
    else:
        print("CUDA not available")

#------------------------------------------------------------------------------
#Parameters

    models = ['LeNet','C4','C8','C16','D4','D8','D16']
    #models = ['LeNet']
    use_entropy = 1
    adjust_entropy = 1
    frequentist = 0
    M = 26 #number of bins for histogram
    scale_graph = 0
    
    for model_to_plot in models:
        print(model_to_plot)
        #csv_file_2 = pd.read_csv(model_to_plot + '_rotations/' + model_to_plot + '_sn_uncertain.csv')
        #csv_file_2 = pd.read_csv(model_to_plot + '_rotations/' + model_to_plot + '_sn_all.csv')
        #csv_file_2 = pd.read_csv(model_to_plot + '_rotations/' + model_to_plot + '_entropy_0.csv')
        csv_file_2 = pd.read_csv(model_to_plot + '_rotations/' + model_to_plot + '_spectral_norm.csv')
        
        #files = [csv_file_0, csv_file_1, csv_file_2]
        files = [csv_file_2]
        equal_count_bins = True
        
    #------------------------------------------------------------------------------
    #Define variables
    
        bins = np.zeros(M)
        bin_centres = np.zeros(M)
        accuracy = np.zeros(M)
        confidence = np.zeros(M)
        uncertainty = np.zeros(M)
        error = np.zeros(M)
        error_errs = np.zeros(M)
        uncertainty_errs = np.zeros(M)
        ECE=0
        UCE=0
        
        error_bins = np.zeros(M)
        error_bin_centres = np.zeros(M)
        
    #Extract data from csv file
    
        target = files[0]["target"].values
        total_datapoints = len(target)
        overlaps = np.empty(total_datapoints)
        
        i=0
        for csv_file in files:
            
            softmax_probs_str = csv_file['softmax prob'].values
            softmax_probs = np.empty([len(softmax_probs_str)])
            
            j=0
            for str in softmax_probs_str:
                temp = str.strip('[')
                softmax_probs[j] = (1 - float(temp.split()[0]))
                j += 1
                
            image_errors = csv_file['classification error %'].values
            
            if use_entropy:
                overlaps = csv_file['mutual information'].values
                
                if adjust_entropy:
                    j=0
                    for y in overlaps:
                        x = np.exp(sc.special.lambertw(-y))
                        overlaps[j] -= (1-x)*np.log(1-x)/np.log(2)
                        j += 1
            else:
                overlaps = csv_file['average overlap'].values
    
            i += 1
            
            
        fr1_err_rate = 0
        fr2_err_rate = 0
        #avg_err_rate = np.mean(image_errors)
        avg_err_rate = 0
        
        fr1_err_rate = (1-np.mean(image_errors[:48]))*100
        fr2_err_rate = (1-np.mean(image_errors[49:]))*100
        avg_err_rate = (1-np.mean(image_errors))*100
        
        
        #overlaps = 0.4+np.log(overlaps)*overlaps
        print(softmax_probs[78])
        #bin_widths_probs = np.zeros(M)
        #bin_widths_uncert = np.zeros(M)
        #error_target = target
        softmax_probs_sorted = softmax_probs[softmax_probs.argsort()]
        accuracy_sorted = (1 - image_errors[softmax_probs.argsort()])
        #print(accuracy_sorted)
        accuracy_target = target[softmax_probs.argsort()]
        overlap_sorted = overlaps[overlaps.argsort()]
        image_errors_sorted = image_errors[overlaps.argsort()]
        #error_errs_sorted = error_errs[overlaps.argsort()]
        
        
    #Bin data
    
        error_classifications = np.zeros(M)
        classifications = np.zeros(M)
        
    #------------------------------------------------------------------------------
        #Equal count bins
         
        bin_count = int(total_datapoints/M)
        i=0
        for i in range(M-1): #create the bins and their min/max values
        
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
        error_bin_centres[M-1] = (1 - error_bins[M-2])/2 + error_bins[M-2] #set centres of the bins
        bin_centres[M-1] = (1 - bins[M-2])/2 + bins[M-2] 
        #print(error_bin_centres)
        
        i = 0
        for b in range(M): # actually bin the data
            
            uncerts = np.zeros(bin_count)
            error_unc = np.zeros(bin_count)
            for j in range(0, bin_count):
                
                confidence[b] += softmax_probs_sorted[j + (b)*bin_count] #running total of confidence
                uncertainty[b] += overlap_sorted[j + (b)*bin_count] #running total of uncertainty
                uncerts[j] = overlap_sorted[j + (b)*bin_count]
                error_unc[j] = image_errors_sorted[j + (b)*bin_count]
                #uncertainty_errs[b] += (overlap_err_sorted[j + (b)*bin_count]/overlap_sorted[j + (b)*bin_count])**2
                
                # if accuracy_target[j + (b)*bin_count] == 1:
                #     classifications[b] += 1 #total classified in bin
                
            uncertainty_errs[i] = np.std(uncerts)
            error_errs[i] = np.std(error_unc)
            
            for j in range(bin_count):
                
                if image_errors_sorted[j + b*bin_count] >= 0.5:
                    error_classifications[b] += 1 - image_errors_sorted[j + b*bin_count]
                else:
                    error_classifications[b] += image_errors_sorted[j + b*bin_count]
                    
                #error_classifications[b] += image_errors_sorted[j + b*bin_count]
                    
                if accuracy_target[j + b*bin_count] == 1:
                    classifications[b] += accuracy_sorted[j + b*bin_count]
                    
                else:
                    classifications[b] += 1 - accuracy_sorted[j + b*bin_count]
                    
                
            accuracy[i] = classifications[i]/bin_count #average over bin
            uncertainty[i] /= bin_count
            
            #uncertainty_errs[i] /= (bin_count/uncertainty[i])
            #uncertainty_errs[i] = np.sqrt(uncertainty_errs[i])
            confidence[i] /= bin_count
            error[i] = error_classifications[i]/bin_count/0.5
            
            
            ECE += (bin_count/total_datapoints)*abs(accuracy[i] - confidence[i])
            UCE += (bin_count/total_datapoints)*abs(error[i] - uncertainty[i])
            
            #Use average value of the bin as the centre
            bin_centres[b] = confidence[b]
            error_bin_centres[b] = uncertainty[b]
    
            i += 1
            
            
        #Calculate class-wise error rates
        # fr1_uncert = np.std(image_errors[0:49])
        # fr2_uncert = np.std(image_errors[50:103])
        # avg_uncert = np.std(image_errors)
        
        
    
    #------------------------------------------------------------------------------
        
    #Format ECE and UCE, then plot graphs
        #print(max(uncertainty))
        ECE = np.format_float_positional(ECE, precision=4)
        UCE = np.format_float_positional(UCE, precision=4)
        
    #Plot ECE graph
        ax = plt.subplot(111)
        plt.plot(bin_centres[:M], accuracy[:M], linewidth=1, marker='.', markersize=10)
        #plt.scatter(bin_centres[:M], accuracy[:M], linewidth=1, marker='x')
        
        x = np.linspace(0, 1.0, 10)
        y = np.linspace(0, 1.0, 10)
        plt.plot(x,y, '--', linewidth=0.7, color='black')
        plt.grid(visible=True)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.left.set_visible(False)
        ax.spines.bottom.set_visible(False)
        
        plt.title("Model Calibration Plot: " + model_to_plot + " Test data")
        # plt.xlim(0,1)
        # plt.ylim(0,1)
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        textstr = ("ECE =  " + ECE)
        props = dict(boxstyle='square', facecolor='silver', alpha=0.5)
        plt.text(0.05, 0.95, textstr, fontsize=14,
        verticalalignment='top', bbox=props)
        plt.show()
        
        
        
    #Plot UCE graph
        ax = plt.subplot(111)
        
        scale_factor = max(uncertainty[-1], error[-1])
        
        #plt.errorbar(error_bin_centres[:M], error[:M], marker='x', linewidth=1, linestyle='')
        if scale_graph:
            plt.plot(uncertainty[:M]/scale_factor, error[:M]/scale_factor, linewidth=1, marker='.', markersize=10)
        else:
            plt.plot(uncertainty[:M], error[:M], linewidth=1, marker='.', markersize=10)
        
        x = np.linspace(0,1,10)
        y = np.linspace(0,1,10)
        plt.plot(x,y, '--', linewidth=0.7, color='black')
        plt.grid(visible=True)
        #plt.xlim(0,1)
        #plt.ylim(0,1)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.left.set_visible(False)
        ax.spines.bottom.set_visible(False)
        plt.title("Model Calibration Plot: " + model_to_plot + " Test data")
        if use_entropy:
            plt.xlabel("Uncertainty (entropy)")
        else:
            plt.xlabel("Uncertainty (overlap index)")
        plt.ylabel("Error")
        textstr = ("UCE =  " + UCE)
        props = dict(boxstyle='square', facecolor='silver', alpha=0.5)
        plt.text(0.05, 0.95, textstr, fontsize=14,
            verticalalignment='top', bbox=props)
        plt.show()
        
    #Output extra information
        
        print('Average Accuracy:', avg_err_rate)
        print('FR I accuracy:', fr1_err_rate)
        print('FR II accuracy:', fr2_err_rate)
    
    
    return    

main()

 