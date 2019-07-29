import cv2

data = cv2.imread('frame2.jpg')
x, y = data.shape[0:2]
print(x,y)

start_x = 0 
for i in range(1):
	end_x = min((int(0.25*x) + start_x),x)
	start_y = 0
	for j in range(4):
		end_y = min((int(0.25*y) + start_y),y)
		data[start_x,start_y:end_y,] = 255 
		data[end_x-1, start_y:end_y,] = 255 
		data[start_x:end_x,start_y,] = 255
		data[start_x:end_x,end_y-1,] = 255 
		start_y = end_y
	start_x = end_x 

cv2.imwrite('frame2-pattern1.jpg',data)