import numpy as np
from skimage import color, exposure, transform, io
import os
import glob
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

import pandas as pd
import h5py
import time

import sys
import argparse
import cv2
#import cv2.cv as cv
print(cv2.__version__)
#инициализация нейронной сети
NUM_CLASSES = 43
IMG_SIZE = 48


def preprocess_img(img):
	# Histogram normalization in v channel (last dimension of HSV format)
	hsv = color.rgb2hsv(img)
	hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
	img = color.hsv2rgb(hsv)

	# Central square crop
	min_side = min(img.shape[:-1])
	centre = img.shape[0]//2, img.shape[1]//2
	img = img[centre[0]-min_side//2:centre[0]+min_side//2,
			  centre[1]-min_side//2:centre[1]+min_side//2,
			  :]

	# Rescale to standard size
	img = transform.resize(img, (IMG_SIZE, IMG_SIZE), mode='constant')

	# Roll color axis (axis -1, last axis) to axis 0
	# img = np.rollaxis(img,-1)  # no need since we use tf format.

	return img


def cnn_model():
	model = Sequential()

	model.add(Conv2D(32, (3, 3), padding='same', activation='relu',
			  input_shape=(IMG_SIZE, IMG_SIZE, 3)))
		# nb_filters (Keras 1) --> filters (Keras 2). Number of filters = 32. 
		# kernel_size[0] = 3, kernel_size[1] = 3.
		# border_mode (Keras 1) --> padding (Keras 2)
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(NUM_CLASSES, activation='softmax'))
	return model

model = cnn_model()
model.load_weights('model.h5')



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread(args["image"])
output_sv = image.copy()
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# detect circles in the image
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 75)
r_max = 0
x_max = 0
y_max = 0

# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
	print(circles)
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		#cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
	#find max radius
	x,y,r = max(circles, key=lambda x: x[2])
	print(x,y,r)
	output_sv = output_sv[(y+r-2*r-5):(y+r+5), (x-r-5):(x-r+2*r+5)]
	# show the output image
	cv2.imwrite('test.png', output_sv)
	#cv2.imshow("output", np.hstack([image, output]))
	#cv2.imshow("cropped", output_sv)
	#cv2.waitKey(10000)


X_test = []

img_path = '/home/anna/project/detect/test.png'

img = io.imread(img_path)

X_test.append(preprocess_img(img))
X_test = np.array(X_test)
Y_pred = model.predict(X_test)
Y_pred_cl = model.predict_classes(X_test)
probab = Y_pred[0][Y_pred_cl[0]]
#Y_pred = np_utils.categorical_probas_to_classes(Y_pred)
print(Y_pred)
print(Y_pred_cl)
print(probab)