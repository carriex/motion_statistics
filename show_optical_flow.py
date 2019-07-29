import numpy as np 
import cv2


frame1 = cv2.imread('frame2.jpg')
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)	#openCV reads in channel as BGR

frame2 = cv2.imread('frame1.jpg')
now = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

hsv = np.zeros_like(frame1) #256 x 341 
hsv[...,1] = 255 # set the second channel = 0

flow = cv2.calcOpticalFlowFarneback(prvs,now, None, 0.5, 3, 15, 3, 5, 1.2, 0)
print(flow.shape)
mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1],angleInDegrees=True)
hsv[...,0] = ang
hsv[...,2] = cv2.normalize(mag, None, 0, 255,cv2.NORM_MINMAX)
bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
cv2.imwrite('frame2-frame1-gray.jpg',gray)


'''
frame1 = cv2.imread('frame2-frame1.jpg')
frame1_c = cv2.imread('frame2-frame1-convert.jpg')

compare = np.asarray([frame1 == frame1_c])
flat_compare = np.ndarray.flatten(compare)
print(len([boolean for boolean in flat_compare if boolean == False ]))
'''