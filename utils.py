import torch
import numpy as np
import cv2



def get_default_device():
	if torch.cuda.is_available():
		return torch.device('cuda:0')
	else:
		return torch.device('cpu')

def write_motion_boundary(frame,mag,ang,title):
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
	cv2.imwrite(title+'.jpg',bgr)


def show_image(img):
	cv2.imshow('img', img)
	k = cv2.waitKey(0)
	if k == 27:
		cv2.destroyAllWindows()




