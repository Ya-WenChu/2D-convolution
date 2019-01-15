from conv import Conv2D
from PIL import Image
import numpy as np
import torch
from torchvision.utils import save_image as sav
import torchvision
import torchvision.transforms as transforms
from PIL import Image

try:  
    img1  = Image.open('1280x720.jpg')
    img2 = Image.open('1920x1080.jpg')  
except IOError: 
    pass


imgToTensor = transforms.Compose([transforms.ToTensor()]) 
input_img_tensor = (imgToTensor(img2))

# Task 1
conv2d = Conv2D(in_channel=3, o_channel=1, kernel_size=3, stride=1, mode ='known')
[Number_of_ops, output_image] = conv2d.forward(input_img_tensor)
print('num of ops is ' , Number_of_ops)

output_img_norm=(((output_image[:,:] - output_image[:,:].min()) / output_image[:,:].ptp()) * 255.0).astype(np.uint8)
output_img_gray = Image.fromarray(output_img_norm)

output_img_gray.save('eeeee.jpg')
#output_img_gray = Image.fromarray(output_image)
#output_img_gray.save('eeeee.jpg')