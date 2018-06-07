import cv2
import numpy as np
import sys
ar = np.zeros(int(sys.argv[2]))
for i in range(int(sys.argv[2])):
	if(i<1):
		continue
	y = 30 
	x = 160
	avg = 0
	num = 0
	Y=20
	X=20
	a = cv2.imread(sys.argv[1]+ "/500_net_G_val/images/input/img_" + str(i) + ".png")
	for x_i in range(X):
		for y_i in range(Y):
			num = num + 1
			avg = avg + (int(a[x+x_i,y+y_i,0]) + int(a[x+x_i,y+y_i,1]) + int(a[x+x_i,y+y_i,2]))/3.0
			a[x+x_i,y+y_i,0] = 0
			a[x+x_i,y+y_i,1] = 0
			a[x+x_i,y+y_i,2] = 255
			# cv2.circle(a,(x+x_i,y+y_i), 1, (0,0,255), -1)
	
	avg = avg/(num*1.0)
	ar[i-1] = avg
	print(str(i) + "," + str(avg))
	# + "			" + str(int(a[x,0])) + "," + str(int(a[y,x,1])) + "," + str(int(a[y,x,2])))
cv2.imwrite("atest_" + str(i) + ".png",a)
file = "pixel_avg"
np.save(file, ar)

