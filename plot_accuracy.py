# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 09:39:51 2022

@author: samhu
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from plots import plot_err_csv
from plots import plot_loss_csv
from plots import plot_loss_multi

def main():
    #csv_file = open('mirabest_lenet.csv')
    #plot_err_csv(csv_file)
    #plot_loss_csv(csv_file)
    plot_loss_multi(dataset='cn',N=40)
    
    return

main()