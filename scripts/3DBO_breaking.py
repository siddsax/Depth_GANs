import yaml
from PIL import Image
import numpy as np
import cv2
# 0848
i = 1
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
	maxim = np.amax(img_d[:,:,0])
	img_d[:,:,0] = 255.0*img_d[:,:,0]/maxim
	img_d[:,:,1] = 255.0*img_d[:,:,1]/maxim
	img_d[:,:,2] = 255.0*img_d[:,:,2]/maxim

	top_left_n = img_n[0:(img_n.shape[0] / 2 - 1),0:(img_n.shape[1] / 2 - 1)];

	top_right_n = img_n[0:(img_n.shape[0] / 2 - 1), (img_n.shape[1] / 2):(img_n.shape[1] - 1)];
	bottom_left_n = img_n[(img_n.shape[0] / 2):(img_n.shape[0] - 1) , 0:(img_n.shape[1]/ 2 - 1)];
	bottom_right_n = img_n[(img_n.shape[0]/2):(img_n.shape[0] - 1), (img_n.shape[1] / 2):(img_n.shape[1] - 1)];

	top_left_d = img_d[0:(img_d.shape[0] / 2 - 1),0:(img_d.shape[1] / 2 - 1)];
	top_right_d = img_d[0:(img_d.shape[0] / 2 - 1), (img_d.shape[1] / 2):(img_d.shape[1] - 1)];
	bottom_left_d = img_d[(img_d.shape[0] / 2):(img_d.shape[0] - 1) , 0:(img_d.shape[1]/ 2 - 1)];
	bottom_right_d = img_d[(img_d.shape[0]/2):(img_d.shape[0] - 1), (img_d.shape[1] / 2):(img_d.shape[1] - 1)];

	vis = np.concatenate((top_left_n, top_left_d), axis=1)
	out = "img_0" + str(i) + ".1.png"
	cv2.imwrite(out, vis );

	vis = np.concatenate((top_right_n, top_right_d), axis=1)
	out = "img_0" + str(i) + ".2.png"
	cv2.imwrite(out, vis );

	vis = np.concatenate((bottom_left_n, bottom_left_d), axis=1)
	out = "img_0" + str(i) + ".3.png"
	cv2.imwrite(out, vis );

	vis = np.concatenate((bottom_right_n, bottom_right_d), axis=1)
	out = "img_0" + str(i) + ".4.png"
	cv2.imwrite(out, vis );
	print(i)
