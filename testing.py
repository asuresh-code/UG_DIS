import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

files_in_use = os.listdir(f"C:/Users/sures/Downloads/test")
file1_in_use = "C:/Users/sures/Downloads/test/mrabd001359.png"
file2_in_use = "C:/Users/sures/Downloads/IF-AFTER-mrabd001359.png"

adv1_image = Image.open(file1_in_use)
adv2_image = Image.open(file2_in_use)

transform = transforms.Compose([
    transforms.PILToTensor()
])

adv1_img_tensor = transform(adv1_image)
adv2_img_tensor = transform(adv2_image)

print(adv1_img_tensor.shape)
print(set(torch.abs(adv1_img_tensor - adv2_img_tensor).flatten().tolist()))

diff1 = torch.mean(torch.abs(adv1_img_tensor - adv2_img_tensor))
diff2 = torch.max(torch.abs(adv1_img_tensor - adv2_img_tensor))

print(diff1)
print(diff2)

print(adv1_img_tensor.shape)
print(adv2_img_tensor.shape)