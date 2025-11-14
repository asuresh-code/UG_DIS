import torch
import torchvision.transforms as transforms
from PIL import Image


image = "../../images_practice/mrabd005680.png"
img = Image.open(image)

transform = transforms.PILToTensor()
img_tensor = transform(img)


grey_image = img_tensor[0]
grey_image = grey_image[None, :, :]

grey_copy = grey_image.clone().detach().to(torch.uint8)
transform = transforms.ToPILImage()
grey_pil = transform(grey_copy)


save = grey_pil.save("grey_image.png")


colour = grey_image.repeat(3,1,1)

color_pil = transform(colour)


save = color_pil.save("rgb_image.png")