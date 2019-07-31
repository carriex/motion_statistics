from __future__ import print_function, division 
import os
import time 
import cv2 
import numpy as np 
from scipy import ndimage
import multiprocessing
import torch 
import torchvision
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader 



class UCF101DataSet(Dataset):
	def __init__(self, framelist_file, v_flow_list_file, u_flow_list_file, clip_len, crop_size,split,transform=None):
		'''
		datalist_file contains the list of frame information e.g. 
		/Users/carriex/git/supervised_training/data/v_ApplyEyeMakeup_g01_c01/ 1 0
		The shape of the return clip is 3 x clip_len x crop_size x crop_size
		'''
		self.datalist = self.get_datalist(framelist_file,v_flow_list_file,u_flow_list_file)  
		self.transform = transform
		self.clip_len = clip_len
		self.crop_size = crop_size
		self.split = split

	def __len__(self):
		return len(self.datalist)

	def __getitem__(self, idx):

		data = self.datalist[idx]
		np_mean = np.load("ucf101_volume_mean_official.npy") 
		frame_data,v_flow_data,u_flow_data = self.load_clip_data(data)
		crop_x, crop_y = self.get_crop_x_y(frame_data)
		frame_data,v_flow_data,u_flow_data = self.crop(frame_data,crop_x,crop_y), self.crop(v_flow_data,crop_x,crop_y), self.crop(u_flow_data,crop_x,crop_y)
		
		if self.split == "training":
			mirror = np.random.randint(0,2)
			if mirror == 0:
				frame_data, v_flow_data, u_flow_data = self.flip(frame_data),self.flip(v_flow_data), self.flip(u_flow_data)

		#v_flow_data = ((v_flow_data * 40.) / 255.) - 20
		#u_flow_data = ((u_flow_data * 40.) / 255.) - 20

		
		label = self.compute_motion_label(v_flow_data, u_flow_data)

		

		clip,label = self.to_tensor(frame_data,label)


		sample = {'clip':clip, 'label':label}

		

		if self.transform:
			sample = self.transform(sample)

		return sample 

	def get_datalist(self,framelist_file,v_flow_list_file,u_flow_list_file):
		framelist = list(open(framelist_file, 'r'))
		v_flow_list = list(open(v_flow_list_file, 'r'))
		u_flow_list = list(open(u_flow_list_file, 'r'))
		datalist = []
		for i in range(len(framelist)):
			frame = framelist[i].strip('\n').split()
			v_flow = v_flow_list[i].strip('\n').split()
			u_flow = u_flow_list[i].strip('\n').split()
			datalist.append((frame,v_flow,u_flow))
		return datalist

	def load_clip_data(self,data):
		frame, v_flow, u_flow = data 
		frame_dir, start_frame = frame[0], int(frame[1])
		v_flow_dir,u_flow_dir = v_flow[0], u_flow[0]
		frame_data, v_flow_data, u_flow_data = [], [], []
		for i in range(self.clip_len):
			frame = self.read_img(frame_dir,start_frame,i)
			frame_data.append(frame)
			if i < (self.clip_len - 1):
				v_flow = self.read_img(v_flow_dir,start_frame,i)
				u_flow = self.read_img(u_flow_dir,start_frame,i)
				v_flow_data.append(v_flow)
				u_flow_data.append(u_flow)
		frame_data = np.array(frame_data).astype(np.uint8)
		v_flow_data = np.array(v_flow_data).astype(np.uint8)
		u_flow_data = np.array(u_flow_data).astype(np.uint8)
		return frame_data, v_flow_data, u_flow_data

	def read_img(self,img_dir,start_frame,i):
		img_path = os.path.join(img_dir, "frame" + "{:06}.jpg".format(start_frame+i))
		img_origin = cv2.imread(img_path)
		img_resize = cv2.resize(img_origin, (171,128))
		img = np.array(img_resize).astype(np.uint8)
		return img


	def get_crop_x_y(self,images):
		x , y = images.shape[1:3]
		crop_size = self.crop_size
		if self.split == 'training':
			crop_x = np.random.randint(0, x-crop_size)
			crop_y = np.random.randint(0, y-crop_size)
		else:
			crop_x = (x - crop_size) // 2 
			crop_y = (y - crop_size) // 2 

		return crop_x, crop_y

	def crop(self,images,crop_x, crop_y):
		# clip - 16 x 128 x 171 x 3 
		# frame - 128 x 171 x 3
		crop_size = self.crop_size
		crop_images = []
		for img in images:
			img = img[crop_x:(crop_x + crop_size), crop_y:(crop_y+crop_size),:]
			crop_images.append(img)
		return np.array(crop_images).astype(np.uint8)



	def flip(self,images):
		flip_images = []
		for img in images:
			img = cv2.flip(img,1)
			flip_images.append(img)
		return np.array(flip_images).astype(np.uint8)


	def to_tensor(self,clip,label):
		return torch.from_numpy(clip.transpose((3,0,1,2))),torch.from_numpy(np.array(label).astype(np.float32))

	def compute_motion_label(self,v_flow_data, u_flow_data):
		''' 
		Mv - 15 x crop_size x crop_size (Applying image gradient on optical flow image)
		Mu - 15 x crop_size x crop_size 

		14 outputs for each input clip:
		max(vl, vo, ul, uo) on p1, p2, p3 (find_max(pattern) return ul=block_id,uo=angle_bin)
		max(Mv), max(Mu)
		return labels 1 x 14 
		
		'''

		v_mag, v_ang, v_frame_motion = self.compute_motion_boundary(v_flow_data,'v-flow-mb') 
		u_mag, u_ang, u_frame_motion = self.compute_motion_boundary(u_flow_data,'u-flow-mb')

		
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




	def compute_motion_boundary(self,flow_data,title):
		'''
		frame - 15 x 112 x 112 
		return - 112 x 112 
		time: 0.07s 
		''' 


		dx = 0 
		dy = 0 
		frame_motion = []

		
		for i, frame in enumerate(flow_data):
			frame = frame[...,0] #gray scale
			#frame = cv2.normalize(frame,None,0,255,cv2.NORM_MINMAX)
			#frame = ((frame * 40) / 255) - 20 
			frame = frame.astype(np.float32)
			x_filter = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
			y_filter = np.array([[-1, -1, -1],[0, 0, 0],[1, 1, 1]])
			frame_dx = ndimage.convolve(frame,x_filter)
			frame_dy = ndimage.convolve(frame,y_filter)
			dx += frame_dx
			dy += frame_dy
			frame_mag, frame_ang = cv2.cartToPolar(frame_dx,frame_dy,angleInDegrees=True)
			#self.write_motion_boundary(frame,frame_mag, frame_ang, title+str(i)+'.jpg')
			frame_motion.append(np.sum(frame_mag))

		mag, ang = cv2.cartToPolar(dx,dy,angleInDegrees=True)
		#self.write_motion_boundary(frame,mag,ang,title+'.jpg')

		return mag, ang, np.array(frame_motion)



	def compute_local_label(self,data):
		'''
		arg - mag_data, ang_data
		Find block with max motion boundary and coresponding dominant angle 
		For maglist
			avg.append(avg(maglist)) --> magnitude 
		max_index = max(avg)'s index 
		max_angle = max(count(anglist[max_index]))

		return ul, uo (block_id and angle_bin)
		'''

		mag_data, ang_data = data 
		mag_data_avg = np.array([np.mean(mag) for mag in mag_data]).astype(np.float32)
		max_mag_idx = np.argmax(mag_data_avg)
		max_mag_data = np.array(mag_data[max_mag_idx]).reshape(-1,)
		max_ang_data = np.array(ang_data[max_mag_idx]).reshape(-1,)
		max_ang_idx = self.get_dominant_orientation(max_mag_data,max_ang_data)



		return max_mag_idx+1, max_ang_idx+1

	def get_dominant_orientation(self,mag_data,ang_data):
		'''
		list of angle data in radian (0, 2*Pi)
		list of mag data corresponding to the type 
		return the index of the angle with largest mag data

		https://www.learnopencv.com/histogram-of-oriented-gradients/

		'''

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
		'''
		argument - list of mag/ang in a 112 x 112 image
		
		return a list of 16 lists of optical flow and magnitude
		'''


		x = data.shape[0]
		y = data.shape[1]
		pattern_data = []

		start_x = 0 
		for i in range(4):
			end_x = min((int(0.25*x) + start_x),x)
			start_y = 0
			for j in range(4):
				end_y = min((int(0.25*y) + start_y),y)
				pattern_data.append(data[start_x:end_x,start_y:end_y].reshape(-1,))
				start_y = end_y
			start_x = end_x 


		return pattern_data


	def pattern_two_old(self,data):
		'''
		argument - list of mag/ang in a clip
		return a list of 4 lists of optical flow and magnitude 
		'''

		x = data.shape[0]
		y = data.shape[1]
		pattern_data = []
		slice_data = []


		for i in range(4):
			start_x =  i * x // 8 
			end_x = min( x, (x - i*x // 8 ))
			start_y = i * y // 8 
			end_y = min( y, (y - i * y // 8)) 
			slice_data.append([data[x][y] for x in range(start_x,end_x) for y in range(start_y,end_y)]) #all, 3/4, 2/1, 4/1 
		for i in range(4):
			if i == 0:
				pattern_data.append(slice_data[3])
			else:
				pattern_data.append([data for data in slice_data[3-i] if data not in slice_data[4-i]])



		return pattern_data


	def pattern_two(self,data):
		'''
		argument - list of mag/ang in a clip
		return a list of 4 lists of optical flow and magnitude 
		'''
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



	def pattern_three(self,data):
		'''
		argument - list of mag/ang in a 112 x 112 image
		return a list of 8 lists of optical flow and magnitude 
		'''
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

	def write_motion_boundary(self,frame,mag,ang,title):
		'''
		H [0, 180] - Different color (largest is red)
		S [0, 255] - Different color intensity 
		V [0, 255] - 0 all black, 255 all white  
		'''

		h, w = frame.shape
		hsv = np.zeros((h,w,3),np.uint8)	#by default it's float64
		hsv[...,2] = 255				#V (255=white)
		hsv[...,0] = ang/2				#H - different color 
		hsv[...,1] = cv2.normalize(mag, None, 0, 255,cv2.NORM_MINMAX)	#S
		bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
		cv2.imwrite(title,bgr)


	def show_image(self,img):
		cv2.imshow('img', img)
		k = cv2.waitKey(0)
		if k == 27:
			cv2.destroyAllWindows()

'''

trainset = UCF101DataSet(framelist_file='list/rgb_train_linux.list', v_flow_list_file='list/v_flow_train_linux.list', u_flow_list_file='list/u_flow_train_linux.list',clip_len=16, crop_size=112,split="training")
trainset = UCF101DataSet(framelist_file='list/rgb_list.list', v_flow_list_file='list/v_flow_list.list', u_flow_list_file='list/u_flow_list.list',clip_len=16, crop_size=112,split="training")


for i, data in enumerate(trainset):
	print(data['label'])
	if i == 9:
		break
'''


class UCF101(Dataset):
	def __init__(self, datalist_file, clip_len, crop_size,split,transform=None):
		'''
		datalist_file contains the list of frame information e.g. 
		/Users/carriex/git/supervised_training/data/v_ApplyEyeMakeup_g01_c01/ 1 0
		The shape of the return clip is 3 x clip_len x crop_size x crop_size
		'''
		self.datalist = self.get_datalist(datalist_file)  
		self.transform = transform
		self.clip_len = clip_len
		self.crop_size = crop_size
		self.split = split

	def __len__(self):
		return len(self.datalist)

	def __getitem__(self, idx):
	
		data = self.datalist[idx]
		frame_dir, start_frame, label = data[0], int(data[1]), data[2]
		np_mean = np.load("ucf101_volume_mean_official.npy") 
		clip = self.load_frames(frame_dir,start_frame)
		clip = self.crop(clip)
		clip = self.random_flip(clip)
		clip,label = self.to_tensor(clip,label)
		sample = {'clip':clip, 'label':label}

	



		if self.transform:
			sample = self.transform(sample)

		return sample 

	def get_datalist(self,datalist_file):
		datalist = list(open(datalist_file, 'r'))
		for i in range(len(datalist)):
			datalist[i] = datalist[i].strip('\n').split()

		return datalist

	def load_frames(self,frame_dir,start_frame):
		clip = []
		for i in range(self.clip_len):
			frame_path = os.path.join(frame_dir, "frame" + "{:06}.jpg".format(start_frame+i))
			frame_origin = cv2.imread(frame_path)
			frame_resize = cv2.resize(frame_origin, (171, 128))
			clip.append(frame_resize)
		clip = np.array(clip)
		return clip

	def crop(self,clip):
		# clip - 16 x 128 x 171 x 3 
		# frame - 128 x 171 x 3 
		x = clip[0].shape[0]
		y = clip[0].shape[1]
		crop_size = self.crop_size
		crop_clip = []
		if self.split == "training":
			crop_x = np.random.randint(0, x - crop_size)
			crop_y = np.random.randint(0, y - crop_size)
		else:
			crop_x = (x - crop_size) // 2 
			crop_y = (y - crop_size) // 2
		for frame in clip:
			frame = frame[crop_x:(crop_x + crop_size), crop_y:(crop_y+crop_size),:]
			crop_clip.append(frame)
		return np.array(crop_clip).astype(np.uint8)


	def normalize(self,clip,np_mean):
		norm_clip = []
		for i in range(len(clip)):
			norm_clip.append(clip[i] - np_mean[i])
		return np.array(norm_clip).astype(np.uint8)

	def random_flip(self,clip):
		flip_clip = []
		mirror = np.random.randint(0,2)
		if self.split == "training":
			for i in range(len(clip)):
				if mirror == 0:
					flip_clip.append(cv2.flip(clip[i], 1))
				else:
					flip_clip.append(clip[i])
		else:
			flip_clip = clip 

		return np.array(flip_clip).astype(np.uint8)


	def to_tensor(self,clip,label):
		return torch.from_numpy(clip.transpose((3,0,1,2))),torch.from_numpy(np.array(label).astype(np.int64))

	




















