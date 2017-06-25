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

import pygame.camera
import pygame.image

img_path = 'test.png'



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

X_test = []

#img_path = '/home/anna/project/GTSRB/Final_Training/Images/00000/00000_00001.ppm'

img = io.imread(img_path)
last_pred = 0
running = True
while running:
	image_new = io.imread(img_path)
	X_test = []
	#time.sleep(5)
	X_test.append(preprocess_img(image_new))
	X_test = np.array(X_test)
	X_test = np.array(X_test)
	Y_pred = model.predict(X_test)
	Y_pred_cl = model.predict_classes(X_test)
	new_pred = Y_pred_cl
	probab = Y_pred[0][Y_pred_cl[0]]
	#Y_pred = np_utils.categorical_probas_to_classes(Y_pred)
	if new_pred != last_pred:	
		#print(Y_pred)
		print(Y_pred_cl)
		#print(probab)
	last_pred = new_pred