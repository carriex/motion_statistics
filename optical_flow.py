import numpy as np 
import cv2
from scipy import ndimage


def compute_motion_boundary(flow_data):
	'''
	frame - 15 x 112 x 112 
	return - 112 x 112 
	''' 
	dx = 0 
	dy = 0 
	frame_motion = []
	
	for frame in flow_data:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
		frame = frame.astype(np.float32)
		x_filter = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
		y_filter = np.array([[-1, -1, -1],[0, 0, 0],[1, 1, 1]])
		frame_dx = ndimage.convolve(frame,x_filter)
		frame_dy = ndimage.convolve(frame,y_filter)
		dx += frame_dx
		dy += frame_dy
		frame_mag, frame_ang = cv2.cartToPolar(frame_dx,frame_dy)
		show_motion_boundary(frame,frame_mag,frame_ang)
		frame_motion.append(np.sum(frame_mag))

	mag, ang = cv2.cartToPolar(dx,dy)
	return mag, ang, frame_motion

def show_motion_boundary(frame,mag,ang):
	h, w = frame.shape
	hsv = np.zeros((h,w,3),np.uint8)	#by default it's float64
	hsv[...,2] = 255
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,1] = cv2.normalize(mag, None, 0, 255,cv2.NORM_MINMAX)
	bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
	cv2.imshow('motion boundary',bgr)
	k = cv2.waitKey(0)
	if k == 27:
		cv2.destroyAllWindows()



frame1 = cv2.imread('optical_flow.jpg')
frame2 = cv2.imread('optical_flow_1.jpg')
frame3 = cv2.imread('optical_flow_3.jpg')[0]
frame5 = cv2.imread('optical_flow_5.jpg')[0]
check = np.ndarray.flatten(np.asarray([ frame3 == frame5 ]))
print(len([boolean for boolean in check if boolean == True ]))

#mag, ang, frame_motion = compute_motion_boundary([frame1])
#mag1, ang1, frame_motion1 = compute_motion_boundary([frame2])




