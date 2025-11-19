import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


tensor = torch.tensor([3,234,3,1])

newt = torch.topk(tensor, 4)

print(newt)

for index in newt.indices:
    print(index)