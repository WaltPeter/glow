import math 
import torch 
from torch import nn 


class GeLU(nn.Module): 
    '''
    Regular GeLU activation. (Fix not available on Pytorch 1.1)
    '''
    def forward(self, x): 
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LightDense(nn.Module): 
    '''
    LightDense layer from paper "LightLayers: Parameter Efficient Dense and 
    Convolutional Layers for Image Classification"
    Note: This module does not include Activation after Dense. 
    '''

    def __init__(self, in_features:int, out_features:int, k:int=8, use_bias:bool=True):
        super().__init__()
        self.k = k 
        self.out_features = out_features 
        self.use_bias = use_bias 

        # Weights 
        self.w1 = nn.Parameter(torch.randn(in_features, k) * 0.1)
        self.w2 = nn.Parameter(torch.randn(k, out_features) *0.1) 

        # Bias 
        if use_bias: 
            self.bias = nn.Parameter(torch.zeros(1, out_features)) 

    def forward(self, x): 
        w1 = self.w1 * math.sqrt(2/self.out_features) 
        w2 = self.w2 * math.sqrt(2/self.out_features) 

        x = torch.matmul(x, w1) 
        x = torch.matmul(x, w2) 

        if self.use_bias: 
            x += self.bias 

        return x 


class LightMLP(nn.Module):
    '''
    Multilayer perceptron.
    
    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : int
        Number of nodes in the hidden layer.
    out_features : int
        Number of output features.
    p : float
        Dropout probability.
    '''

    def __init__(self, in_features:int, hidden_features:int, out_features:int, use_bias:bool=True, k:int=8, p:float=0.):
        super().__init__()
        self.fc1 = LightDense(in_features, hidden_features, k=k, use_bias=use_bias)
        self.act = GeLU()
        self.fc2 = LightDense(hidden_features, out_features, k=k, use_bias=use_bias) 
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x) 
        x = self.act(x) 
        x = self.drop(x) 
        x = self.fc2(x)
        x = self.drop(x) 
        return x 


class MlpBlock(nn.Module):
    '''
    Multilayer perceptron. 

    Parameters
    ----------
    dim : int 
        Number of input and output features. 
    mlp_dim : int 
        Number of nodes in the hidden layer. 
    '''

    def __init__(self, dim, mlp_dim=None, lite=False): 
        super().__init__()
        self.Linear = nn.Linear if not lite else LightDense 
        mlp_dim = dim if mlp_dim is None else mlp_dim
        self.linear_1 = self.Linear(dim, mlp_dim)
        self.activation = GeLU()
        self.linear_2 = self.Linear(mlp_dim, dim)

    def forward(self, x):
        x = self.linear_1(x) 
        x = self.activation(x) 
        x = self.linear_2(x) 
        return x