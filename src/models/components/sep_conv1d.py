import torch
from torch import nn
from torch import optim
from masksembles.torch import Masksembles2D, Masksembles1D
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader
import pdb

class _SepConv1d(nn.Module):
    """A simple separable convolution implementation.
    
    The separable convlution is a method to reduce number of the parameters 
    in the deep learning network for slight decrease in predictions quality.
    """
    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv1d(ni, ni, kernel, stride, pad, groups=ni),
            nn.Conv1d(ni, no, kernel_size=1)
        )

    def forward(self, x):
        return self.model(x)
		
class SepConv1d(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.
    
    The module adds (optionally) activation function and dropout 
    layers right after a separable convolution layer.
    """
    def __init__(self, hparams: dict):
        super().__init__()
        
        #assert drop is None or (0.0 < drop < 1.0)
        layers = [_SepConv1d(hparams['num_in'], hparams['num_out'], hparams['kernel'], hparams['stride'], hparams['pad'])]
        if hparams['activ']:
            layers.append(hparams['activ'])
        if hparams['batch_norm']:
            layers.append(nn.BatchNorm1d(hparams['num_out']))
        if hparams['dropout'] is not None:
            layers.append(nn.Dropout(hparams['dropout']))

        self.model = nn.Sequential(*layers)
        
    def forward(self, x): 
        return self.layers(x)