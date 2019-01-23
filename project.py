# Image Segmentation With Expectation Maximization
# https: // github.com/agileshaw/Project-Euler

import numpy as np
from sklearn.mixture import GaussianMixture as mixture
import matplotlib.pyplot as plt
import cv2 as cv
import argparse

#read image from path
parser = argparse.ArgumentParser(
    description='Image Segmentation With Expectation Maximization')
parser.add_argument('--input', help='Path to image.',
                    default='/Users/agileshaw/Desktop/dog.png')
args = parser.parse_args()
path = cv.samples.findFile(args.input)
img = cv.imread(path)

#convert image to greyscale and blur it for noise removal
grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
grey = cv.blur(grey, (3, 3))
grey = cv.GaussianBlur(grey, (3, 3), 0, 0)

#get the rows and cols info from the input image
rows, cols = grey.shape

#convert the grey image to float array with 1-dimension
floatImg = np.float32(grey)
print(floatImg)
array = np.reshape(floatImg, (-1, 1))
print(array)

#train the data with Expectation Maximazation on Gaussian Mixture Model
gmm = mixture(n_components=2, covariance_type='full')
gmm = gmm.fit(array)

#predict the labels for the data samples by trained model
cluster = gmm.predict(array)
cluster = np.reshape(cluster, (rows, cols))
print(cluster)

#seperate foreground and background images
lv1 = img.copy()
lv2 = img.copy()
for i in range(rows):
    for j in range(cols):
        if cluster[i][j] == 0:
            lv1[i][j] = 0
        else:
            lv2[i][j] = 0

#output the original, foreground, background images
plt.subplot(131), plt.imshow(img)
plt.axis('off')
plt.subplot(132), plt.imshow(lv1)
plt.axis('off')
plt.subplot(133), plt.imshow(lv2)
plt.axis('off')
plt.show()
