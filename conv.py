import numpy as np

class Conv2D:
	def __init__(self, in_channel, o_channel, kernel_size, stride, mode):
		self.in_channel = in_channel
		self.o_channel = o_channel
		self.kernel_size = kernel_size
		self.stride = stride
		self.mode = mode

		self.k1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]] )
		self.k2 = np.array([[-1,  0,  1], [-1, 0, 1], [-1, 0, 1]])
		self.k3 = np.array([[1,  1,  1], [1, 1, 1], [1, 1, 1]])
		self.k4 = np.array([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
		self.k5 = np.array([[-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1]])

		pass

	def forward(self,input_image):
		self.input = np.asarray(input_image)
		a,b,c = self.input.shape
		print(a,b,c)
		self.image_arr= np.hstack(self.input)
		print(self.image_arr.shape)
		# Task 1
		if self.o_channel == 1:
			self.kernel = [self.k1]

		# Task 2
		elif self.o_channel == 2:
			self.kernel = [self.k4, self.k5]

		# Task 3
		elif self.o_channel == 3:
			self.kernel = [self.k1, self.k2, self.k3]

		summation = 0
		product = 0

		for index in range(len(self.kernel)):
			cur_kernel = self.kernel[index]
			row = len(self.image_arr)
			col = len(self.image_arr[0])
			print('row',row)
			print('col',col)
			r_size = 1+int((row-self.kernel_size)/self.stride)
			c_size = 1+int((col-self.kernel_size)/self.stride)
			# create empty array
			new_image = np.zeros((r_size, c_size), dtype=int)
			print('r-size',r_size)
			print('c-size',c_size)
			for i in range(r_size):
				for j in range(c_size):
					# perform calc
					s = 0
					# slice smaller image out from bigger image
					#F = np.array(self.image_arr[ (i * self.kernel_size) : (i * self.kernel_size+self.kernel_size) , (j * self.kernel_size) : (j * self.kernel_size+self.kernel_size)])
					F = self.image_arr[ (i * self.stride) : (i * (self.stride) +self.kernel_size) , (j * self.stride) : (j * self.stride+self.kernel_size)]
					F = np.fliplr(np.flipud(F))
					for n in range(len(F)):
						for m in range(len(F[0])):
							#print(F[n,m]*cur_kernel[n,m])
							s += F[n,m]*cur_kernel[n,m]
							summation += 1
							product += 1
							#print(s)
					new_image[i,j] = s


			self.image_arr = (new_image)
			num_of_ops = summation+product

			# flip the image


			return (num_of_ops, self.image_arr)