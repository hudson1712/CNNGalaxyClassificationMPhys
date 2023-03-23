import torch

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchsummary import summary
import pandas as pd
import os
from torchvision.io import read_image
from torchvision.io import ImageReadMode

import numpy as np
import csv
import glob
from PIL import Image
import random as rd
import matplotlib as plt
import umap
import astropy.stats
from sklearn.preprocessing import StandardScaler

from models import VanillaLeNet, CNSteerableLeNet, DNSteerableLeNet, GPLeNet, DN_GPLeNet
from utils import *
from plots import *
from FRDEEP import FRDEEPF
from MiraBest import *
#from MiraBest_F import *
from galaxy_mnist import *


beta = 2.5 #Temperature for Platt scaling
T = 3/(np.pi**2)
T = 1
use_ood = 0
use_training = 1

# -----------------------------------------------------------------------------
# extract information from config file:
    
use_entropy = True
gp = True

vars = parse_args()
config_dict, config = parse_config(vars['config'])

if gp:
    batch_size = 104
else:
    batch_size = 1    
    
epochs         = config_dict['training']['epochs']
imsize         = config_dict['training']['imsize']
nclass         = config_dict['training']['num_classes']

nrot           = config_dict['model']['nrot']

outdir         = config_dict['metrics']['outputdir']
modelfiles     = glob.glob(outdir+"/*.pt")

config         = vars['config'].split('/')[-1][:-4]
csvfile        = 'latent_values.csv'
feature_file   = 'feature_space.csv'

# -----------------------------------------------------------------------------
# check if gpu is available:

use_cuda = torch.cuda.is_available()
#use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print("Device: ", device)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# load data set:

crop     = transforms.CenterCrop(imsize)
pad      = transforms.Pad((0, 0, 1, 1), fill=0)
totensor = transforms.ToTensor()
normalise= transforms.Normalize((config_dict['data']['datamean'],), (config_dict['data']['datastd'],))

transform = transforms.Compose([
    crop,
    pad,
    transforms.RandomRotation(360, interpolation=Image.BILINEAR, expand=False),
    totensor,
    normalise,
])


test_data = locals()[config_dict['data']['dataset']](config_dict['data']['datadir'], train=False, download=True, transform=transform)
if use_training:
    test_data = locals()[config_dict['data']['dataset']](config_dict['data']['datadir'], train=True, download=True, transform=transform)
    batch_size=729
    
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

# norm = test_data[:]/255
# print(norm.mean())
# print(norm.std())


N = len(test_loader)
targ = np.zeros(N, dtype=np.int8)

if use_ood:
    
        
    # 224 pixel images
    optical_galaxy_dataset = GalaxyMNISTHighrez(
        root='OOD_data/optical_galaxies',
        download=True,
        train=False  # by default, or set False for test set
    )
    optical_images, opt_labels = optical_galaxy_dataset._load_data()



    class CustomImageDataset(Dataset):
        def __init__(self, img_dir='OOD_data/optical_galaxies/GalaxyMNISTHighrez/raw/test_dataset', transform=None, target_transform=None):
            self.img_labels = np.zeros(50)
            self.img_dir = 'OOD_data/images'
            self.transform = transform
            self.target_transform = target_transform
    
        def __len__(self):
            return 50
    
        def __getitem__(self, idx):
            img_files = glob.glob(self.img_dir+"/*.jpg")
            img_path = img_files[idx]
            #image = read_image(img_path, mode = ImageReadMode.GRAY)
            
            rand_idx = rd.randint(0, 1950)
            image = optical_images[idx][0]/255
                    
            
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
                
                
           
            label = 2    
            print(torch.max(image), torch.mean(image), torch.std(image))
            
            #image = image.detach().cpu().numpy()
            return image, 2
    
    toimage  = transforms.ToPILImage()    
    crop     = transforms.CenterCrop(imsize)
    pad      = transforms.Pad((0, 0, 1, 1), fill=0)
    totensor = transforms.ToTensor()
    
    #normalise= transforms.Normalize(ood_mean, ood_std)
    normalise= transforms.Normalize((config_dict['data']['datamean'],), (config_dict['data']['datastd'],))
    
    transform = transforms.Compose([
        toimage,
        crop,
        #transforms.RandomRotation(360, interpolation=Image.BILINEAR, expand=False),
        pad,
        totensor,
        normalise,
    ])
    
    ood_data = CustomImageDataset(transform=transform)
    
    #test_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_size, shuffle=False)
    
    N = len(test_loader)
    targ = np.zeros(N, dtype=np.int8)
    
    print(len(test_data.filenames))
    
    
    
    combined_data = torch.utils.data.ConcatDataset([test_data, ood_data])
    test_loader = torch.utils.data.DataLoader(combined_data, batch_size=154, shuffle=False)
    


# -----------------------------------------------------------------------------
# specify model:

model = locals()[config_dict['model']['base']](1, nclass, imsize+1, kernel_size=5, N=nrot).to(device)

# -----------------------------------------------------------------------------

metrics=[]

# load saved model:
if use_cuda:
    #model.load_state_dict(torch.load(modelfiles[6]),strict=0)
    state_dict = torch.load(modelfiles[0], map_location='cpu')
    model.load_state_dict(state_dict, strict=0)
    model.to(device)
else:
    model.load_state_dict(torch.load(modelfiles[0], map_location=torch.device('cpu')),strict=0)


rows = ['target', 'softmax prob', 'latent_parameter_1', 'latent_parameter_2', 'classification error %', 'variance', 'predictive entropy', 'average entropy', 'mutual information']

            
with open(csvfile, 'a', newline="") as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(rows)

print('There are ',N,' data files')

if gp:
    # get straight prediction using full test set:
    model.eval()
    for batch_idx, (data, labels) in enumerate(test_loader):
        
        data, labels = data.to(device), labels.to(device)
        output, gp_covmat, latent_features = model(data)
        #output, gp_covmat = model(data)
        latent_features = latent_features.detach().cpu().numpy()

        sngp_variance = torch.linalg.diagonal(gp_covmat)[:, None].detach().cpu().numpy()
    
    np.savetxt("feature_space.csv", latent_features, delimiter=",")
    np.savetxt("feature_labels.csv", labels.detach().cpu().numpy(), delimiter=",")
    
    latent_param_1 = output[:,0].detach().cpu().numpy()
    latent_param_2 = output[:,1].detach().cpu().numpy()
    
    p = F.softmax(output,dim=1)[:].detach().cpu().numpy()
    
    entropy = calc_GPentropy(p)
    
    for j in range(len(data)):
        _results = [labels[j].item(), p[j], latent_param_1[j], latent_param_2[j], 0, sngp_variance[j,0], entropy[j]]
        if use_ood:
            if j>103:
                _results = [labels[j].item(), p[j], latent_param_1[j], latent_param_2[j], 0, sngp_variance[j,0], entropy[j]]
        
        with open(csvfile, 'a', newline="") as f_out:
            writer = csv.writer(f_out, delimiter=',')
            writer.writerow(_results)
                
else:
    for i in range(0,N):
    
        subset_indices = [i] # select your indices here as a list
        subset = torch.utils.data.Subset(test_data, subset_indices)
        testloader_ordered = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
        data, target = next(iter(testloader_ordered))
        
        data = data.to(device)
        
        # get straight prediction:
        model.eval()
        x = model(data)
        
        latent_param_1 = x[0,0].detach().cpu().numpy()
        latent_param_2 = x[0,1].detach().cpu().numpy()
        p = F.softmax(x,dim=1)[0].detach().cpu().numpy()
        
        av_overlap, std_overlap, class_error, p_entropy, a_entropy, mi = fr_latent_space_test(model, data, target, i, device, beta)

        # create output row:
        _results = [target[0].item(), p, latent_param_1, latent_param_2, class_error, av_overlap, p_entropy, a_entropy, mi]
        if use_ood:
            _results = [2, p, latent_param_1, latent_param_2, class_error, av_overlap, p_entropy, a_entropy, mi]
        
        with open(csvfile, 'a', newline="") as f_out:
            writer = csv.writer(f_out, delimiter=',')
            writer.writerow(_results)

#print(gp_covmat)
covmat = (gp_covmat.detach().cpu().numpy())/torch.max(gp_covmat).detach().cpu().numpy() 

plt.matshow(covmat);
plt.colorbar()
plt.show()



