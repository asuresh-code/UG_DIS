import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

b = torch.Tensor([4,2,31,9])

og = torch.Tensor([0.2,0.3,0.4,0.1])
tens = torch.max(og)
factor = 2.0/tens

print(b + og*torch.Tensor(factor))