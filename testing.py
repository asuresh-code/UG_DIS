import torch
import torchvision.transforms as transforms
import PIL as Image

img_tensor = torch.tensor([[[2, 4], [5, 2]],[[2, 1], [42, 2]],[[4,4], [5, 9]]])
print(img_tensor)
print(img_tensor.shape)

grey_image = img_tensor[0]
grey_image = grey_image[None, :, :]
print(grey_image.shape)
print(grey_image)

colour = grey_image.repeat(3,1,1)

print(colour.shape)
print(colour)