from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os.path
import sys
# from __future__ import print_function  # Only needed for Python 2
import sys
def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	var = np.var(imageA.astype("float") - imageB.astype("float"))
	return err

def compare_images(imageA, imageB, title):
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)
f = open('all_losses', 'a')
num = 0
err1 = 0.0
ssim1 = 0.0
arr_ind = np.load(sys.argv[1] + ".npy")
for i in range(1,int(sys.argv[2])):
	print(i)
        original = cv2.imread(sys.argv[1] + "/500_net_G_val/images/input/img_" + str(int(arr_ind[i-1]))+".png")
	#print(sys.argv[1] + "/500_net_G_val/images/input/img_" + str(arr_ind[i-1])+".png")
	output = cv2.imread(sys.argv[1] + "/500_net_G_val/images/output/img_" + str(i)+".png")
	#print(sys.argv[1] + "/500_net_G_val/images/input/img_" + str(i)+".png")
        if os.path.isfile(sys.argv[1] + "/500_net_G_val/images/input/img_" + str(i)+".png"):
		original = cv2.cvtColor(original[0:255,0:255,:], cv2.COLOR_BGR2GRAY)
		output = cv2.cvtColor(output[0:255,0:255,:], cv2.COLOR_BGR2GRAY)
		num = num +1
		err1=err1 + mse(original, output)/255.0
		ssim1=ssim1 + ssim(original, output)
		#print(ssim1)		
# print("For our method MSE = " + str(err1) + " variance is " + str(var1)  + " SSIM = " +  str(ssim1)) 
err1 = err1/num
ssim1 = ssim1/num
f.write(sys.argv[1] + "," + str(err1) + "," + str(ssim1) + "\n")
f.close()

