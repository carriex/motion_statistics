from __future__ import print_function, division 
import os
import cv2 
import numpy as np 
from scipy import ndimage
import torch 
from torch.utils.data import Dataset, DataLoader 
from utils import write_motion_boundary, show_image
import constant 



class UCF101DataSet(Dataset):
	def __init__(self, framelist_file, clip_len, crop_size, split, v_flow_list_file=None, u_flow_list_file=None):
		"""
		UCF101Dataset for video class label or motion statistic label 
		(if the v_flow_list_file and u_flow_list_file is present)
		"""

		self.framelist = self.get_datalist(framelist_file)
		self.v_flow_list_file = v_flow_list_file
		self.u_flow_list_file = u_flow_list_file
		self.clip_len = clip_len
		self.crop_size = crop_size
		self.split = split

		if v_flow_list_file and u_flow_list_file:
			self.v_flow_list = self.get_datalist(v_flow_list_file)
			self.u_flow_list = self.get_datalist(u_flow_list_file)

	def __len__(self):
		return len(self.framelist)

	def __getitem__(self, idx):

		frame_data = self.load_clip_data(self.framelist[idx], self.clip_len)
		crop_x, crop_y = self.get_crop_x_y(frame_data)
		frame_data= self.crop(frame_data, crop_x, crop_y)

		mirror = 1
		
		if self.split == "training":
			mirror = np.random.randint(0, 2)
			if mirror == 0:
				frame_data = self.flip(frame_data)

		if self.v_flow_list_file and self.u_flow_list_file:
			v_flow_data = self.load_clip_data(self.v_flow_list[idx], self.clip_len-1)
			u_flow_data = self.load_clip_data(self.u_flow_list[idx], self.clip_len-1)
			v_flow_data = self.crop(v_flow_data, crop_x, crop_y)
			u_flow_data = self.crop(u_flow_data, crop_x, crop_y)
			if mirror == 0:
				v_flow_data, u_flow_data = self.flip(v_flow_data), self.flip(u_flow_data)
			label = self.compute_motion_label(v_flow_data, u_flow_data)
		else:
			label = self.read_frame_label(self.framelist[idx])

		

		clip,label = self.to_tensor(frame_data, label)


		sample = {'clip':clip, 'label':label}

		

		return sample 

	def get_datalist(self, framelist_file):
		"""
		Return the dir and file info from the list file
		"""
		framelist = list(open(framelist_file, 'r'))
		datalist = []
		for frame in framelist:
			datalist.append(frame.strip('\n').split())
		return datalist

	def load_clip_data(self, framelist_data, num_of_frame):
		"""
		Return a list of image data with number of the frame specific
		For flow data, num of frame = clip_len - 1
		"""
		frame_dir, start_frame = framelist_data[0], int(framelist_data[1])
		frame_data= []
		for i in range(num_of_frame):
			frame = self.read_img(frame_dir, start_frame, i)
			frame_data.append(frame)
		frame_data = np.array(frame_data).astype(np.uint8)
		return frame_data

	def read_frame_label(self, framelist_data):
		return framelist_data[2]


	def read_img(self, img_dir, start_frame, i):
		"""
		Read the image from the path and resize it 
		"""
		img_path = os.path.join(img_dir, "frame" + "{:06}.jpg".format(start_frame+i))
		img_origin = cv2.imread(img_path)
		img_resize = cv2.resize(img_origin, (constant.RESIZE_H, constant.RESIZE_W))
		img = np.array(img_resize).astype(np.uint8)
		return img


	def get_crop_x_y(self, images):
		"""
		Return and (x,y) value for the left most point of the crop 
		Crop is random during training, otherwise a center crop
		"""
		x , y = images.shape[1:3]
		crop_size = self.crop_size
		if self.split == 'training':
			crop_x = np.random.randint(0, x-crop_size)
			crop_y = np.random.randint(0, y-crop_size)
		else:
			crop_x = (x - crop_size) // 2 
			crop_y = (y - crop_size) // 2 

		return crop_x, crop_y

	def crop(self, images, crop_x, crop_y):
		"""
		Crop all images in the list 
		"""

		crop_size = self.crop_size
		crop_images = []
		for img in images:
			img = img[crop_x:(crop_x + crop_size), crop_y:(crop_y+crop_size),:]
			crop_images.append(img)
		return np.array(crop_images).astype(np.uint8)



	def flip(self, images):
		"""
		Flip all images in the clip horizontally 
		"""
		flip_images = []
		for img in images:
			img = cv2.flip(img, 1)
			flip_images.append(img)
		return np.array(flip_images).astype(np.uint8)


	def to_tensor(self,clip,label):
		return torch.from_numpy(clip.transpose((3,0,1,2))),torch.from_numpy(np.array(label))

	def compute_motion_label(self, v_flow_data, u_flow_data):
		"""
		Args: List of optical flow data between two frames in a video clip
		- 
		Return: a list of 14 motion statistics label across one sample of v and u flow data (16 frames in a video clip, 15 motion boundary)
		- The block of largest average motion boundary magnitude based on three different patterns:
		(v_block_id_pattern_1, v_block_id_pattern_2, v_block_id_pattern_3, 
		u_block_id_pattern_1, u_block_id_pattern_2, u_block_id_pattern3)
		- The dominanting angle in the above block (calculated with histogram of orient)
		(v_angle_bin_pattern_1, v_angle_bin_pattern_2, v_angle_bin_pattern_3, 
		u_angle_bin_pattern_1, u_angle_bin_pattern_2, u_angle_bin_pattern3)
		- The frame with the maximum motion boundary among the 15 motion boundary on u and v flow data respectively 
		(max_v_idx, max_u_idx)
		"""

		v_mag, v_ang, v_frame_motion = self.compute_motion_boundary(v_flow_data, 'v-flow-mb') 
		u_mag, u_ang, u_frame_motion = self.compute_motion_boundary(u_flow_data, 'u-flow-mb')

		
		p_f = [self.pattern_one, self.pattern_two, self.pattern_three]
		v_pattern_data = [] 
		u_pattern_data = []



		for f in p_f:
			v_pattern_data.append((f(v_mag),f(v_ang)))
			u_pattern_data.append((f(u_mag),f(u_ang)))

		
		label = []
		
		for i in range(len(v_pattern_data)):
			v_block_id, v_angle_bin = self.compute_local_label(v_pattern_data[i])
			u_block_id, u_angle_bin = self.compute_local_label(u_pattern_data[i])
			label.extend([v_block_id,v_angle_bin,u_block_id,u_angle_bin])


		max_v_idx = np.argmax(v_frame_motion) + 1 	
		max_u_idx = np.argmax(u_frame_motion) + 1
		label.extend([max_v_idx,max_u_idx])


		

		return label




	def compute_motion_boundary(self, flow_data, title):
		"""
		Args:
		- List of flow data between two frames in a video clip 
		Return:
		- the magnitude and angle of the sum of the motion boundary of all the frames in a video clip
		- the list of magnitude and angle of motion boundary between each two frames in a video clip
		"""


		d_x = 0 
		d_y = 0 
		frame_motion = []

		
		for i, frame in enumerate(flow_data):
			frame = frame[...,0] #gray scale
			frame = frame.astype(np.float32)
			x_filter = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
			y_filter = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
			frame_dx = ndimage.convolve(frame,x_filter)
			frame_dy = ndimage.convolve(frame,y_filter)
			d_x += frame_dx
			d_y += frame_dy
			frame_mag, _ = cv2.cartToPolar(frame_dx, frame_dy, angleInDegrees=True)
			frame_motion.append(np.sum(frame_mag))

		mag, ang = cv2.cartToPolar(d_x, d_y, angleInDegrees=True)

		return mag, ang, np.array(frame_motion)



	def compute_local_label(self,data):
		"""
		Args: 
		- a list of magnitude and angle in a block defined by a pattern
		Return:
		- the block_id with largest magnitude and the id of corresponding dominant orientation 
		Both ID starts from 1  
		"""

		mag_data, ang_data = data 
		mag_data_avg = np.array([np.mean(mag) for mag in mag_data]).astype(np.float32)
		max_mag_idx = np.argmax(mag_data_avg)
		max_mag_data = np.array(mag_data[max_mag_idx]).reshape(-1,)
		max_ang_data = np.array(ang_data[max_mag_idx]).reshape(-1,)
		max_ang_idx = self.get_dominant_orientation(max_mag_data, max_ang_data)



		return max_mag_idx+1, max_ang_idx+1

	def get_dominant_orientation(self, mag_data, ang_data):
		"""
		Args:
		- List of magnitude data
		- List of corresponding angle data in degree 
		Return 
		- Index of the angle with largest mag data computed in the manner of https://www.learnopencv.com/histogram-of-oriented-gradients/
		Index starts from 0 

		"""

		bin_size = 8 
		bin_value = 360 / bin_size 
		mag_bin = [0] * 8

		for i,ang in enumerate(ang_data):
			lower_bin = int(ang // bin_value) % 8
			upper_bin = (lower_bin + 1) % bin_size 
			lower_portion = 1 - ((ang % bin_value) / bin_value)
			mag_bin[lower_bin]+=mag_data[i]*lower_portion 
			mag_bin[upper_bin]+=mag_data[i]*(1-lower_portion)

		max_idx = np.argmax(mag_bin)


		return max_idx



	def pattern_one(self,data):
		"""
		Args:
		- List of a N X N data 
		Return: 
		- List of 16 lists containing the data based on pattern 1  
		"""


		x = data.shape[0]
		y = data.shape[1]
		pattern_data = []

		start_x = 0 
		for i in range(4):
			end_x = min((int(0.25*x) + start_x), x)
			start_y = 0
			for j in range(4):
				end_y = min((int(0.25*y) + start_y), y)
				pattern_data.append(data[start_x:end_x,start_y:end_y].reshape(-1,))
				start_y = end_y
			start_x = end_x 


		return pattern_data


	def pattern_two(self, data):
		"""
		Args:
		- List of a N X N data 
		Return: 
		- List of 4 lists containing the data based on pattern 1  
		"""
		x = data.shape[0]
		y = data.shape[1]
		pattern_data = []
		slice_data = []
		indices = []



		for i in range(4):
			start_x =  i * x // 8 
			end_x = min( x, (x - i*x // 8 ))
			start_y = i * y // 8 
			end_y = min( y, (y - i * y // 8)) 
			indices.append((start_x, end_x, start_y, end_y))
		for i in range(4):
			start_x, end_x, start_y, end_y = indices[3-i]
			if i == 0:
				pattern_data.append(data[start_x:end_x, start_y:end_y].reshape(-1,))
			else:
				last_start_x, last_end_x, last_start_y, last_end_y = indices[4-i]
				this_data_block = list(data[start_x:last_start_x,start_y:end_y].reshape(-1,)) + list(data[last_end_x:end_x,start_y:end_y].reshape(-1,)) + list(data[last_start_x:last_end_x,start_y:last_start_y].reshape(-1,)) + list(data[last_start_x:last_end_x,last_end_y:end_y].reshape(-1,)) 
				pattern_data.append(this_data_block)


		return pattern_data



	def pattern_three(self, data):
		"""
		Args:
		- List of a N X N data 
		Return: 
		- List of 8 lists containing the data based on pattern 1  
		"""
		x = data.shape[0]
		y = data.shape[1]
		pattern_data = []

		start_x = 0 
		for i in range(2):
			start_y = 0 
			end_x = min(int(0.5*x) + start_x, x)
			for j in range(2):
				end_y = min((int(0.5*x) + start_y),y)
				indices = [(idx_x, idx_y) for idx_x in range(start_x, end_x) for idx_y in range(start_y, end_y)]
				if start_x == start_y:
					pattern_data.append([data[idx_x][idx_y] for (idx_x, idx_y) in indices if idx_x > idx_y])
					pattern_data.append([data[idx_x][idx_y] for (idx_x, idx_y) in indices if idx_x <= idx_y])
				else:
					pattern_data.append([data[idx_x][idx_y] for (idx_x, idx_y) in indices if idx_x <= y - idx_y])
					pattern_data.append([data[idx_x][idx_y] for (idx_x, idx_y) in indices if idx_x > y - idx_y])
				start_y = end_y
			start_x = end_x

		return pattern_data


'''
trainset = UCF101DataSet(framelist_file='list/rgb_list.list', v_flow_list_file='list/v_flow_list.list', u_flow_list_file='list/u_flow_list.list',clip_len=16, crop_size=112,split="training")
trainset = UCF101DataSet(framelist_file='list/rgb_list.list', clip_len=16, crop_size=112,split="training")


for i, data in enumerate(trainset):
	print(data['label'])
	if i == 9:
		break
'''



	




















