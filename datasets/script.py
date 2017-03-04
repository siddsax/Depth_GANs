import yaml
from PIL import Image
import numpy as np
import cv2
# 0848

for i in range(1,848):
	if i < 10:
	  path_1 = "KinectColor/" + "img_000" + str(i) + ".png"
	  path_2 = "RegisteredDepthData/" + "img_000" + str(i) + "_abs_smooth.png"
	elif i < 100:
	  path_1 = "KinectColor/" + "img_00" + str(i) + ".png"
	  path_2 = "RegisteredDepthData/" + "img_00" + str(i) + "_abs_smooth.png"
	else:
	  path_1 = "KinectColor/" + "img_0" + str(i) + ".png"
	  path_2 = "RegisteredDepthData/" + "img_0" + str(i) + "_abs_smooth.png"

	img_n = cv2.imread(path_1)
	img_d = cv2.imread(path_2)
	height,width, channels = img_d.shape

	width = width/2
	height = height/2
	img_n = cv2.resize(img_n,(width, height))
	img_d = cv2.resize(img_d,(width, height))

	vis = np.concatenate((img_n, img_d), axis=1)
	out = "img_0" + str(i) + ".png"
	cv2.imwrite(out, vis );
