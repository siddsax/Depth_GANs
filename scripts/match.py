import cv2
import numpy as np
import sys


ar = np.zeros(int(sys.argv[2]))
arr2 = np.load("pixel_avg.npy")

s = sys.argv[1]
for i in range(int(sys.argv[2])):
	# if(i<121):
		# continue
	y = 215
	x = 205
	avg = 0
	num = 0
	Y=20
	X=20
# for i in range(7):
	a = cv2.imread(s+ "/500_net_G_val/images/output/img_" + str(i+1) + ".png")
	for x_i in range(X):
		for y_i in range(Y):
			num = num + 1
			avg = avg + (int(a[x+x_i,y+y_i,0]) + int(a[x+x_i,y+y_i,1]) + int(a[x+x_i,y+y_i,2]))/3.0
			a[x+x_i,y+y_i,0] = 0
			a[x+x_i,y+y_i,1] = 0
			a[x+x_i,y+y_i,2] = 255
			# cv2.circle(a,(x+x_i,y+y_i), 1, (0,0,255), -1)
	avg = avg/(num*1.0)
	# cv2.imwrite("atest_output_" + str(i) + ".png",a)
	
	idx = (np.abs(arr2-avg)).argmin()
	ar[i] = idx
	print(str(i) + "," + str(avg) + "," + str(idx))
cv2.imwrite("atest_2_" + str(i) + ".png",a)
np.save(sys.argv[1],ar)
