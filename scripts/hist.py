import cv2
import numpy as np
from matplotlib import pyplot as plt 
img = cv2.imread("Layer47Filters_s1_l10_e500_a0_t1.jpg")
hist_1 = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(hist_1,label="learned")
#plt.hist(img.ravel(),256,[0,256],label='learned')
#plt.savefig("hist.png")

img = cv2.imread("Initialized_Layer47Filters_s1_l10_e500_a0_t1.jpg")
hist_2 = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(hist_2,label="initialized")
#plt.hist(img.ravel(),256,[0,256],label='Initialized')
#plt.savefig("Initialized_hist.png")
plt.legend(loc='best')
plt.savefig("Final.png")
