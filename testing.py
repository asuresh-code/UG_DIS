import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

files_in_use = os.listdir(f"C:/Users/sures/Downloads/test")
file1_in_use = "C:/Users/sures/Downloads/freq1mrabd023312.png"
file2_in_use = "C:/Users/sures/Downloads/freq2mrabd023312.png"


file3_in_use = "C:/Users/sures/Downloads/mrabd023312.png"

adv1_image = Image.open(file3_in_use)

transform = transforms.Compose([
    transforms.PILToTensor()
])

adv1_img_tensor = transform(adv1_image)

ntan1 = adv1_img_tensor.clone()[0]
ntan2 = adv1_img_tensor.clone()[1]
ntan3 = adv1_img_tensor.clone()[2]

newt1 = ntan1.repeat(3,1,1)
newt2 = ntan2.repeat(3,1,1)
newt3 = ntan3.repeat(3,1,1)

transform = transforms.ToPILImage()
img = transform(newt1.to(torch.uint8))
sv = img.save("newt1.png")

img = transform(newt2.to(torch.uint8))
sv = img.save("newt2.png")

img = transform(newt3.to(torch.uint8))
sv = img.save("newt3.png")