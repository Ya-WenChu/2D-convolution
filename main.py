from conv import Conv2D
from PIL import Image
import numpy as np
import torch
from torchvision.utils import save_image as sav
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

try:  
	img0 = Image.open('checkerboard.png')
	img1 = Image.open('1280x720.jpg')
	#img2 = Image.open('1920x1080.jpg')  
except IOError: 
    pass


imgToTensor = transforms.Compose([transforms.ToTensor()]) 
checkerboard = imgToTensor(img0)#(imgToTensor(img1),imgToTensor(img2))
colorImage = imgToTensor(img1)




# Task 1
o_channel = 1
conv2d = Conv2D(in_channel=1, o_channel=1, kernel_size=3, stride=1, mode ='known', task=1)
[Number_of_ops, output_image] = conv2d.forward(checkerboard)
print('num of ops for Task 1 is ' , Number_of_ops)

output_image = output_image.numpy()
for i in range(o_channel):
	output_img_norm=(((output_image[i,:,:] - output_image[i,:,:].min()) / output_image[i,:,:].ptp()) * 255.0).astype(np.uint8)
	output_img_gray = Image.fromarray(output_img_norm)
	output_img_gray.save('task1.jpg')

# Task 2
o_channel = 1
conv2d = Conv2D(in_channel=1, o_channel=1, kernel_size=3, stride=1, mode ='known', task=2)
[Number_of_ops, output_image] = conv2d.forward(checkerboard)
print('num of ops for Task 2 is ', Number_of_ops)

output_image = output_image.numpy()
for i in range(o_channel):
	output_img_norm=(((output_image[i,:,:] - output_image[i,:,:].min()) / output_image[i,:,:].ptp()) * 255.0).astype(np.uint8)
	output_img_gray = Image.fromarray(output_img_norm)
	output_img_gray.save('task2.jpg')

# Task 3
o_channel = 1
conv2d = Conv2D(in_channel=1, o_channel=1, kernel_size=3, stride=2, mode='known', task=3)
[Number_of_ops, output_image] = conv2d.forward(checkerboard)
print('num of ops for Task 3 is ', Number_of_ops)

output_image = output_image.numpy()
for i in range(o_channel):
	output_img_norm=(((output_image[i,:,:] - output_image[i,:,:].min()) / output_image[i,:,:].ptp()) * 255.0).astype(np.uint8)
	output_img_gray = Image.fromarray(output_img_norm)
	output_img_gray.save('task3.jpg')

# Task 4
o_channel = 1
conv2d = Conv2D(in_channel=1, o_channel=1, kernel_size=5, stride=2, mode='known', task=4)
[Number_of_ops, output_image] = conv2d.forward(checkerboard)
print('num of ops for Task 4 is ', Number_of_ops)

output_image = output_image.numpy()
for i in range(o_channel):
	output_img_norm=(((output_image[i,:,:] - output_image[i,:,:].min()) / output_image[i,:,:].ptp()) * 255.0).astype(np.uint8)
	output_img_gray = Image.fromarray(output_img_norm)
	output_img_gray.save('task4.jpg')

# Task 5
o_channel = 1
conv2d = Conv2D(in_channel=1, o_channel=1, kernel_size=5, stride=2, mode='known', task=5)
[Number_of_ops, output_image] = conv2d.forward(checkerboard)
print('num of ops for Task 5 is ', Number_of_ops)

output_image = output_image.numpy()
for i in range(o_channel):
	output_img_norm=(((output_image[i,:,:] - output_image[i,:,:].min()) / output_image[i,:,:].ptp()) * 255.0).astype(np.uint8)
	output_img_gray = Image.fromarray(output_img_norm)
	output_img_gray.save('task5.jpg')

# Task 6
o_channel = 1
conv2d = Conv2D(in_channel=3, o_channel=1, kernel_size=3, stride=1, mode='known', task=6)
[Number_of_ops, output_image] = conv2d.forward(colorImage)
print('num of ops for Task 6 is ', Number_of_ops)

output_image = output_image.numpy()
for i in range(o_channel):
	output_img_norm=(((output_image[i,:,:] - output_image[i,:,:].min()) / output_image[i,:,:].ptp()) * 255.0).astype(np.uint8)
	output_img_gray = Image.fromarray(output_img_norm)
	output_img_gray.save('task6.jpg')

# Task 7
o_channel = 2
conv2d = Conv2D(in_channel=3, o_channel=2, kernel_size=3, stride=1, mode='known', task=7)
[Number_of_ops, output_image] = conv2d.forward(colorImage)
print('num of ops for Task 7 is ', Number_of_ops)

output_image = output_image.numpy()
for i in range(o_channel):
	output_img_norm=(((output_image[i,:,:] - output_image[i,:,:].min()) / output_image[i,:,:].ptp()) * 255.0).astype(np.uint8)
	output_img_gray = Image.fromarray(output_img_norm)
	output_img_gray.save('task7_%s.jpg'%(i+1))


# Part B Task 8
time_taken = []
for i in range(0,10):
	print(i)
	o_channel = 2**i
	conv2d = Conv2D(in_channel=3, o_channel=o_channel, kernel_size=3, stride=1, mode='rand', task=8)
	start_time = time.time()
	[Number_of_ops, output_image] = conv2d.forward(colorImage)
	time_elapse = time.time() - start_time
	time_taken.append(time_elapse)
	print(time_elapse)

plt.plot([1,2,4,8,16,32,64,128,256], time_taken)
plt.show()


# Part C Task 9
k_size_arr = [3, 5, 7, 9, 11]
num_op = []
for i in range(0,5):
	k_size = k_size_arr[i]
	print(k_size)
	conv2d = Conv2D(in_channel=3, o_channel=3, kernel_size=k_size, stride=1, mode='rand', task=9)
	[Number_of_ops, output_image] = conv2d.forward(colorImage)
	print(Number_of_ops)
	num_op.append(Number_of_ops)

plt.plot(k_size_arr, num_op)
plt.show()

