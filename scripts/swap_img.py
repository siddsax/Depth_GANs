# Run on python 3 for CSEProj 145
import numpy as np
import cv2

top = 461
for i in range(1,top):
# Load an color image in grayscalei
	try:
		name = 'img_' + str(i) + '.png'
		img = cv2.imread(name)
	except:
		continue
	if img is not None:
		#print(np.shape(img))
		a = img[:,0:255,:]
		b = img[:,255:510,:]
		c = np.concatenate((b,a),axis=1)
		cv2.imwrite(name,c)
	else:
		print(i)
