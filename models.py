import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from torch import Tensor

from e2cnn import gspaces
from e2cnn import nn as e2nn

import utils
import copy

        
# -----------------------------------------------------------------------------

class VanillaLeNet(nn.Module):
    def __init__(self, in_chan, out_chan, imsize, kernel_size=5, N=None):
        super(VanillaLeNet, self).__init__()
        
        z = 0.5*(imsize - 2)
        z = int(0.5*(z - 2))
        
        self.mask = utils.build_mask(imsize, margin=1)

        self.conv1 = nn.Conv2d(in_chan, 6, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size, padding=1)
        self.fc1   = nn.utils.spectral_norm(nn.Linear(16*z*z, 120))
        self.fc2   = nn.utils.spectral_norm(nn.Linear(120, 84))
        self.fc3   = nn.utils.spectral_norm(nn.Linear(84, out_chan), n_power_iterations=5)
        self.drop  = nn.Dropout(p=0.5)
        
        # dummy parameter for tracking device
        self.dummy = nn.Parameter(torch.empty(0))
        
    def loss(self,p,y):
        
        # check device for model:
        device = self.dummy.device
        
        # p : softmax(x)
        loss_fnc = nn.NLLLoss().to(device=device)
        loss = loss_fnc(torch.log(p),y)
        
        return loss
     
    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        return
        
    def forward(self, x):
        
        # check device for model:
        device = self.dummy.device
        mask = self.mask.to(device=device)
        
        x = x*mask
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size()[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        
        #print(x)
        x = self.fc3(x)
        #print(x)
        
    
        return x, np.empty(50)

# -----------------------------------------------------------------------------
#Implement SNGP with random features

def RandomFeatureLinear(i_dim, o_dim, bias=True, require_grad=False):
    m = nn.Linear(i_dim, o_dim, bias)
    # https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/models/bert_sngp.py
    nn.init.normal_(m.weight, mean=0.0, std=0.05)
    # freeze weight
    m.weight.requires_grad = require_grad
    if bias:
        nn.init.uniform_(m.bias, a=0.0, b=2. * np.pi)
        # freeze bias
        m.bias.requires_grad = require_grad
    #print(len(m.weight))
    return m


class GPLeNet(nn.Module):
    def __init__(self, in_chan, out_chan, imsize, kernel_size=5, N=None, 
                 hidden_size = 84,
                 gp_kernel_scale=1,
                 num_inducing=1024,
                 gp_output_bias=0.,
                 layer_norm_eps=1e-12,
                 scale_random_features=True,
                 n_power_iterations=5,
                 normalize_input=True,
                 gp_cov_momentum=-1,
                 gp_cov_ridge_penalty=1,
                 num_classes=2,
                 device='cuda'):
        super(GPLeNet, self).__init__()
        
        z = 0.5*(imsize - 2)
        z = int(0.5*(z - 2))
        #hidden_size = 16*z*z
        
        self.mask = utils.build_mask(imsize, margin=1)

        self.conv1 = nn.Conv2d(in_chan, 6, kernel_size, padding=1)
        self.bn1   = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size, padding=1)
        self.bn2   = nn.BatchNorm2d(16)
        self.fc1   = nn.utils.spectral_norm(nn.Linear(16*z*z, 120), n_power_iterations=1)
        self.fc2   = nn.utils.spectral_norm(nn.Linear(120, hidden_size), n_power_iterations=1)
        self._random_feature = RandomFeatureLinear(hidden_size, num_inducing)
        self._gp_output_layer =  nn.utils.spectral_norm(nn.Linear(num_inducing, num_classes, bias=False),n_power_iterations=1)
        
        # dummy parameter for tracking device
        self.dummy = nn.Parameter(torch.empty(0))
        
        self.gp_input_scale = 1. / np.sqrt(gp_kernel_scale)
        self.gp_feature_scale = np.sqrt(2. / float(num_inducing))
        self.gp_output_bias = gp_output_bias
        self.scale_random_features = scale_random_features
        self.normalize_input = normalize_input
        self.gp_cov_ridge_penalty = gp_cov_ridge_penalty
        self.gp_cov_momentum = gp_cov_momentum
        self._gp_input_normalize_layer = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self._gp_output_bias = torch.tensor([self.gp_output_bias] * num_classes).to(device)
        

        # Laplace Random Feature Covariance
        # Posterior precision matrix for the GP's random feature coefficients.
        self.initial_precision_matrix = (self.gp_cov_ridge_penalty * torch.eye(num_inducing).to(device))
        self.precision_matrix = torch.nn.Parameter(copy.deepcopy(self.initial_precision_matrix), requires_grad=False)
        
        
    def gp_layer(self, gp_inputs, update_cov=True):
        # Supports lengthscale for custom random feature layer by directly
        # rescaling the input.
        if self.normalize_input:
            gp_inputs = self._gp_input_normalize_layer(gp_inputs)

        gp_feature = self._random_feature(gp_inputs)
        # cosine
        gp_feature = torch.cos(gp_feature)

        if self.scale_random_features:
            gp_feature = gp_feature * self.gp_input_scale

        # Computes posterior center (i.e., MAP estimate) and variance.
        gp_output = self._gp_output_layer(gp_feature) + self._gp_output_bias

        if update_cov:
            # update precision matrix
            self.update_cov(gp_feature)
        return gp_feature, gp_output

    def reset_cov(self):
        self.precision_matrix = torch.nn.Parameter(copy.deepcopy(self.initial_precision_matrix), requires_grad=False)

    def update_cov(self, gp_feature):
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L346
        batch_size = gp_feature.size()[0]
        precision_matrix_minibatch = torch.matmul(gp_feature.t(), gp_feature)
        # Updates the population-wise precision matrix.
        if self.gp_cov_momentum > 0:
            # Use moving-average updates to accumulate batch-specific precision
            # matrices.
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (
                    self.gp_cov_momentum * self.precision_matrix +
                    (1. - self.gp_cov_momentum) * precision_matrix_minibatch)
        else:
            # Compute exact population-wise covariance without momentum.
            # If use this option, make sure to pass through data only once.
            precision_matrix_new = self.precision_matrix + precision_matrix_minibatch
        #self.precision_matrix.weight = precision_matrix_new
        self.precision_matrix = torch.nn.Parameter(precision_matrix_new, requires_grad=False)


    def compute_predictive_covariance(self, gp_feature):
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L403
        # Computes the covariance matrix of the feature coefficient.
        feature_cov_matrix = torch.linalg.inv(self.precision_matrix)

        # Computes the covariance matrix of the gp prediction.
        cov_feature_product = torch.matmul(feature_cov_matrix, gp_feature.t()) * self.gp_cov_ridge_penalty
        gp_cov_matrix = torch.matmul(gp_feature, cov_feature_product)
        return gp_cov_matrix
        
        
    def loss(self,p,y):
        
        # check device for model:
        device = self.dummy.device
        
        # p : softmax(x)
        loss_fnc = nn.NLLLoss().to(device=device)
        #print(p)
        loss = loss_fnc(torch.log(p),y)
        
        return loss
     
    def forward(self, x, update_cov: bool = True, return_gp_cov: bool = True):
        
        T=3/np.pi**2
        T=1
        
        device = self.dummy.device
        mask = self.mask.to(device=device)

        x = x*mask
        
        x = self.bn1(F.relu(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.bn2(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size()[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
            
        gp_feature, gp_output = self.gp_layer(x, update_cov=update_cov)
        gp_cov_matrix = self.compute_predictive_covariance(gp_feature)
        
        
        
        if not (self.training):
            sngp_variance = torch.sqrt(torch.linalg.diagonal(gp_cov_matrix)[:, None])
            
            gp_output[:] /= torch.sqrt(1 + T*sngp_variance[:])
            
        
        print(gp_cov_matrix)
        
        return gp_output, gp_cov_matrix
    
#-------------------------------------------------------------------------------
    

# -----------------------------------------------------------------------------

class CNSteerableLeNet(nn.Module):
    def __init__(self, in_chan, out_chan, imsize, kernel_size=5, N=8):
        super(CNSteerableLeNet, self).__init__()
        
        z = 0.5*(imsize - 2)
        z = int(0.5*(z - 2))
        
        self.r2_act = gspaces.Rot2dOnR2(N)
        
        in_type = e2nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        
        out_type = e2nn.FieldType(self.r2_act, 6*[self.r2_act.regular_repr])
        self.mask = e2nn.MaskModule(in_type, imsize, margin=1)
        self.conv1 = e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False)
        self.relu1 = e2nn.ReLU(out_type, inplace=True)
        self.pool1 = e2nn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)

        in_type = self.pool1.out_type
        out_type = e2nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr])
        self.conv2 = e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False)
        self.relu2 = e2nn.ReLU(out_type, inplace=True)
        self.pool2 = e2nn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)
        
        self.gpool = e2nn.GroupPooling(out_type)

        self.fc1   = nn.Linear(16*z*z, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.utils.spectral_norm(nn.Linear(84, out_chan))
        
        self.drop  = nn.Dropout(p=0.5)
        
        # dummy parameter for tracking device
        self.dummy = nn.Parameter(torch.empty(0))
        
        
    def loss(self,p,y):
        
        # check device for model:
        device = self.dummy.device
        
        # p : softmax(x)
        loss_fnc = nn.NLLLoss().to(device=device)
        loss = loss_fnc(torch.log(p),y)
        
        return loss
     
    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        return
      
      
    def forward(self, x):
        
        x = e2nn.GeometricTensor(x, self.input_type)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.gpool(x)
        x = x.tensor
        
        x = x.view(x.size()[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
    
        return x

# -----------------------------------------------------------------------------


class DNSteerableLeNet(nn.Module):
    def __init__(self, in_chan, out_chan, imsize, kernel_size=5, N=8):
        super(DNSteerableLeNet, self).__init__()
        
        z = 0.5*(imsize - 2)
        z = int(0.5*(z - 2))
        
        self.r2_act = gspaces.FlipRot2dOnR2(N)
        
        in_type = e2nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        
        out_type = e2nn.FieldType(self.r2_act, 6*[self.r2_act.regular_repr])
        self.mask = e2nn.MaskModule(in_type, imsize, margin=1)
        self.conv1 = e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False)
        self.relu1 = e2nn.ReLU(out_type, inplace=True)
        self.pool1 = e2nn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)

        in_type = self.pool1.out_type
        out_type = e2nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr])
        self.conv2 = e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False)
        self.relu2 = e2nn.ReLU(out_type, inplace=True)
        self.pool2 = e2nn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)
        
        self.gpool = e2nn.GroupPooling(out_type)

        self.fc1   = nn.Linear(16*z*z, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.utils.spectral_norm(nn.Linear(84, out_chan))
        
        self.drop  = nn.Dropout(p=0.5)
        
        # dummy parameter for tracking device
        self.dummy = nn.Parameter(torch.empty(0))
        
    def loss(self,p,y):
        
        # check device for model:
        device = self.dummy.device
        
        # p : softmax(x)
        loss_fnc = nn.NLLLoss().to(device=device)
        loss = loss_fnc(torch.log(p),y)
        
        return loss
     
    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        return
 
    def forward(self, x):
        
        x = e2nn.GeometricTensor(x, self.input_type)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.gpool(x)
        x = x.tensor
        
        x = x.view(x.size()[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
    
        return x

# -----------------------------------------------------------------------------

class DN_GPLeNet(nn.Module):
    def __init__(self, in_chan, out_chan, imsize, kernel_size=5, N=8, 
                 hidden_size = 84,
                 gp_kernel_scale=0.1,
                 num_inducing=1024,
                 gp_output_bias=0.,
                 layer_norm_eps=1e-12,
                 scale_random_features=True,
                 n_power_iterations=5,
                 normalize_input=True,
                 gp_cov_momentum=0.999,
                 gp_cov_ridge_penalty=1e-3,
                 num_classes=2,
                 device='cuda'):
        super(DN_GPLeNet, self).__init__()
        
        z = 0.5*(imsize - 2)
        z = int(0.5*(z - 2))
        #hidden_size = 16*z*z
        
        self.mask = utils.build_mask(imsize, margin=1)
        
        self.r2_act = gspaces.FlipRot2dOnR2(N)
        
        in_type = e2nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        
        out_type = e2nn.FieldType(self.r2_act, 6*[self.r2_act.regular_repr])
        self.mask = e2nn.MaskModule(in_type, imsize, margin=1)
        self.conv1 = e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False)
        self.relu1 = e2nn.ReLU(out_type, inplace=True)
        self.pool1 = e2nn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)

        in_type = self.pool1.out_type
        out_type = e2nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr])
        self.conv2 = e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False)
        self.relu2 = e2nn.ReLU(out_type, inplace=True)
        self.pool2 = e2nn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)
        
        self.gpool = e2nn.GroupPooling(out_type)

        self.bn1   = nn.BatchNorm2d(6)
        self.bn2   = nn.BatchNorm2d(16)
        self.fc1   = nn.utils.spectral_norm(nn.Linear(16*z*z, 120), n_power_iterations=1)
        self.fc2   = nn.utils.spectral_norm(nn.Linear(120, hidden_size), n_power_iterations=1)
        self._random_feature = RandomFeatureLinear(hidden_size, num_inducing)
        self._gp_output_layer =  nn.utils.spectral_norm(nn.Linear(num_inducing, num_classes, bias=False),n_power_iterations=1)
        
        
        # dummy parameter for tracking device
        self.dummy = nn.Parameter(torch.empty(0))
        
        self.gp_input_scale = 1. / np.sqrt(gp_kernel_scale)
        self.gp_feature_scale = np.sqrt(2. / float(num_inducing))
        self.gp_output_bias = gp_output_bias
        self.scale_random_features = scale_random_features
        self.normalize_input = normalize_input
        self.gp_cov_ridge_penalty = gp_cov_ridge_penalty
        self.gp_cov_momentum = gp_cov_momentum
        self._gp_input_normalize_layer = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self._gp_output_bias = torch.tensor([self.gp_output_bias] * num_classes).to(device)
        

        # Laplace Random Feature Covariance
        # Posterior precision matrix for the GP's random feature coefficients.
        self.initial_precision_matrix = (self.gp_cov_ridge_penalty * torch.eye(num_inducing).to(device))
        self.precision_matrix = torch.nn.Parameter(copy.deepcopy(self.initial_precision_matrix), requires_grad=False)
        
        
    def gp_layer(self, gp_inputs, update_cov=True):
        if self.normalize_input:
            gp_inputs = self._gp_input_normalize_layer(gp_inputs)

        gp_feature = self._random_feature(gp_inputs)
        gp_feature = torch.cos(gp_feature)

        if self.scale_random_features:
            gp_feature = gp_feature * self.gp_input_scale

        gp_output = self._gp_output_layer(gp_feature) + self._gp_output_bias

        if update_cov:
            self.update_cov(gp_feature)
        return gp_feature, gp_output

    def reset_cov(self):
        self.precision_matrix = torch.nn.Parameter(copy.deepcopy(self.initial_precision_matrix), requires_grad=False)

    def update_cov(self, gp_feature):

        batch_size = gp_feature.size()[0]
        precision_matrix_minibatch = torch.matmul(gp_feature.t(), gp_feature)

        if self.gp_cov_momentum > 0:

            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (
                    self.gp_cov_momentum * self.precision_matrix +
                    (1. - self.gp_cov_momentum) * precision_matrix_minibatch)
        else:
            precision_matrix_new = self.precision_matrix + precision_matrix_minibatch

        self.precision_matrix = torch.nn.Parameter(precision_matrix_new, requires_grad=False)


    def compute_predictive_covariance(self, gp_feature):
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L403
        # Computes the covariance matrix of the feature coefficient.
        feature_cov_matrix = torch.linalg.inv(self.precision_matrix)

        # Computes the covariance matrix of the gp prediction.
        cov_feature_product = torch.matmul(feature_cov_matrix, gp_feature.t()) * self.gp_cov_ridge_penalty
        gp_cov_matrix = torch.matmul(gp_feature, cov_feature_product)
        return gp_cov_matrix
        
        
    def loss(self,p,y):
        
        # check device for model:
        device = self.dummy.device
        
        # p : softmax(x)
        loss_fnc = nn.NLLLoss().to(device=device)
        #print(p)
        loss = loss_fnc(torch.log(p),y)
        
        return loss
     
    def forward(self, x, update_cov: bool = True, return_gp_cov: bool = True):
        
        T=3/np.pi**2
        T=1
        
        x = e2nn.GeometricTensor(x, self.input_type)
        
        x = self.conv1(x)
        x = self.bn1(self.relu1(x))
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(self.relu2(x))
        x = self.pool2(x)
        
        x = self.gpool(x)
        x = x.tensor
        
        x = x.view(x.size()[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
            
        gp_feature, gp_output = self.gp_layer(x, update_cov=update_cov)
        gp_cov_matrix = self.compute_predictive_covariance(gp_feature)
        
        if not (self.training):
            sngp_variance = torch.sqrt(torch.linalg.diagonal(gp_cov_matrix)[:, None])
            
            gp_output[:] /= torch.sqrt(1 + T*sngp_variance[:])
            
        
        return gp_output, gp_cov_matrix



class DNRestrictedLeNet(nn.Module):
    def __init__(self, in_chan, out_chan, imsize, kernel_size=5, N=8):
        super(DNRestrictedLeNet, self).__init__()
        
        z = 0.5*(imsize - 2)
        z = int(0.5*(z - 2))
        
        self.r2_act = gspaces.FlipRot2dOnR2(N)
        
        in_type = e2nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        
        out_type = e2nn.FieldType(self.r2_act, 6*[self.r2_act.regular_repr])
        self.mask = e2nn.MaskModule(in_type, imsize, margin=1)
        self.conv1 = e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False)
        self.relu1 = e2nn.ReLU(out_type, inplace=True)
        self.pool1 = e2nn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)

        self.gpool = e2nn.GroupPooling(out_type)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size, padding=1)
        
        self.fc1   = nn.Linear(16*z*z, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, out_chan)
        
        self.drop  = nn.Dropout(p=0.5)
        
        # dummy parameter for tracking device
        self.dummy = nn.Parameter(torch.empty(0))
        
    def loss(self,p,y):
        
        # check device for model:
        device = self.dummy.device
        
        # p : softmax(x)
        loss_fnc = nn.NLLLoss().to(device=device)
        loss = loss_fnc(torch.log(p),y)
        
        return loss
     
    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        return
 
    def forward(self, x):
        
        x = e2nn.GeometricTensor(x, self.input_type)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.gpool(x)
        x = x.tensor
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size()[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
    
        return x

# -----------------------------------------------------------------------------

class HMTNet(nn.Module):
    
    """
        This network has been taken directly from "transfer learning for radio galaxy classification"
        https://arxiv.org/abs/1903.11921
        """
    
    def __init__(self, in_chan, out_chan, imsize, kernel_size=11, N=None):
        super(HMTNet,self).__init__()
        
        
        self.conv1 = nn.Conv2d(in_channels=in_chan,out_channels=6,kernel_size=(11,11),padding=5,stride=1)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5),padding=2,stride=1)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=24,kernel_size=(3,3),padding=1,stride=1)
        self.conv4 = nn.Conv2d(in_channels=24,out_channels=24,kernel_size=(3,3),padding=1,stride=1)
        self.conv5 = nn.Conv2d(in_channels=24,out_channels=16,kernel_size=(3,3),padding=1,stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=5, stride=5)
        self.bnorm1 = nn.BatchNorm2d(6)
        self.bnorm2 = nn.BatchNorm2d(16)
        self.bnorm3 = nn.BatchNorm2d(24)
        self.bnorm4 = nn.BatchNorm2d(24)
        self.bnorm5 = nn.BatchNorm2d(16)
        
        self.fc1 = nn.Linear(400,256) #channel_size * width * height
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,out_chan)
        
        self.dropout = nn.Dropout(p=0.5)
        
        # dummy parameter for tracking device
        self.dummy = nn.Parameter(torch.empty(0))

    
    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


    def loss(self,p,y):
        
        # check device for model:
        device = self.dummy.device
        
        # p : softmax(x)
        loss_fnc = nn.NLLLoss().to(device=device)
        loss = loss_fnc(torch.log(p),y)
        
        return loss

    
    def forward(self, x):
        
        x  = F.relu(self.conv1(x))
        x = self.bnorm1(x)
        x = self.pool1(x)
        
        x  = F.relu(self.conv2(x))
        x = self.bnorm2(x)
        x = self.pool2(x)
        
        x  = F.relu(self.conv3(x))
        x = self.bnorm3(x)
        
        x  = F.relu(self.conv4(x))
        x = self.bnorm4(x)
        
        x  = F.relu(self.conv5(x))
        x = self.bnorm5(x)
        x = self.pool3(x)
        
        x = x.view(x.size()[0], -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        v = self.fc3(x)

        return v


# -----------------------------------------------------------------------------

