# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 14:34:02 2022
Samuel Hudson
"""

import torch

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchsummary import summary

import numpy as np
import csv
from PIL import Image

from models import VanillaLeNet, CNSteerableLeNet, DNSteerableLeNet, DNRestrictedLeNet
from utils import *
from FRDEEP import FRDEEPF
from MiraBest import MBFRConfident
#from MingoLoTSS import MLFR

def main():
    
    if(torch.cuda.is_available()):
        print('CUDA on')
    else:
        print("CUDA not available")
    return

main()