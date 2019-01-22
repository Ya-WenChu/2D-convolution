import numpy as np
import torch
import PIL

class Conv2D:
	def __init__(self, in_channel, o_channel, kernel_size, stride, mode, task):
		self.in_channel = in_channel
		self.o_channel = o_channel
		self.kernel_size = kernel_size
		self.stride = stride
		self.mode = mode
		self.task = task

		self.k1 = torch.Tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]] )
		self.k2 = torch.Tensor([[-1,  0,  1], [-1, 0, 1], [-1, 0, 1]])
		self.k3 = torch.Tensor([[1,  1,  1], [1, 1, 1], [1, 1, 1]])
		self.k4 = torch.Tensor([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
		self.k5 = torch.Tensor([[-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1]])
		self.k6 = torch.Tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]] )
		self.k7 = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
		self.k8 = [[-1, 0, -1], [-2, 0, 2], [-1, 0, 1]]

		pass

	def forward(self,input_image):
		self.input = input_image 
		[channel, img_height, img_width] = self.input.size()
		#print(channel,img_height,img_width)
		self.image_arr= self.input


		# Task 1
		if self.task == 1:
			self.kernel = [self.k1]
			
		# Task 2
		elif self.task == 2:
			self.kernel = [self.k2]

		# Task 3
		elif self.task == 3:
			self.kernel = [self.k3]

		# Task 4
		elif self.task == 4:
			self.kernel = [self.k4]

		# Task 5
		elif self.task == 5:
			self.kernel = [self.k5]

		# Task 6
		elif self.task == 6:
			self.kernel = [self.k6, self.k6, self.k6]

		# Task 7
		elif self.task == 7:
			self.kernel = [torch.Tensor([self.k7, self.k7, self.k7]), torch.Tensor([self.k8, self.k8, self.k8])]

		# Task 8 and Task 9
		if self.mode == 'rand':
			self.kernel = torch.randn(self.kernel_size, self.kernel_size,self.kernel_size)
		#kernel1 = torch.stack([self.k1 for i in range(self.in_channel)]) 
		summation = 0
		product = 0

		for k in self.kernel:
			#print(k)
			k = torch.stack([k for i in range(self.in_channel)])

		row = img_height
		col = img_width

		r_size = 1+int((row-self.kernel_size)/self.stride)
		c_size = 1+int((col-self.kernel_size)/self.stride)

		# create empty array
		new_image = torch.zeros(self.o_channel, r_size, c_size) #np.zeros((r_size, c_size), dtype=int)
		#print(new_image.size())


		for index in range(self.o_channel):
			#k = self.kernel[index]
			k = self.kernel
			for i in range(r_size):
				for j in range(c_size):
					# perform calc
					# slice smaller image out from bigger image
					F = self.image_arr[:,(i * self.stride) : (i * (self.stride) +self.kernel_size) , (j * self.stride) : (j * self.stride+self.kernel_size)]
					#print('k size', k.size())
					#print(F.size())
					temp = torch.mul(k,F)

					summation += self.kernel_size**self.kernel_size -1
					product += self.kernel_size**self.kernel_size
					new_image[index,i,j] = temp.sum()

			#if(self.o_channel==1):
		#		self.image_arr = torch.unsqueeze(new_image,0)
		#	else:
			#self.image_arr = new_image

			num_of_ops = summation+product


		return (num_of_ops, new_image)