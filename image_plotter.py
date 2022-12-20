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
from plots import *
from FRDEEP import FRDEEPF
from MiraBest import MBFRConfident
#from MingoLoTSS import MLFR

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# extract information from config file:

   

for i in range(0,104):
    number = str(i)
    img_name = ('galaxies/radio_galaxy_' + number + '.png')
    plot_image(i, img_name, 'magma')