import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is: " + str(device))

class PiecewiseReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.RelU()
    def forward(self, x):
        return x + self.f((x - 1) / 2.0) + self.f(-(x + 1) / 2.0)
        
class Interleaved(nn.Module):
    def __init__(self, ):
        super().__init__()
        
