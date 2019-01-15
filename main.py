from conv import Conv2D
from PIL import Image
import numpy as np
import imageio
try:  
    img1  = Image.open('1280x720.jpg')
    img2 = Image.open('1920x1080.jpg')  
except IOError: 
    pass

# Task 1
conv2d = Conv2D(in_channel=3, o_channel=1, kernel_size=3, stride=1, mode ='known')
[Number_of_ops, output_image] = conv2d.forward(img2)
print('num of ops is ' , Number_of_ops)


imageio.imwrite('eeeee.jpg', output_image)
#output_img_gray = Image.fromarray(output_image)
#output_img_gray.save('eeeee.jpg')