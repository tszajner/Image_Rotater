
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy


import scipy
import numpy as np
import argparse

import os
from glob import glob
import urllib
import time
import pprint

import imghdr


class Image_Rotater():

	def __init__(self):

		image_size=32
		self.image_size = image_size
		# Same network definition as before
		
		img_prep = ImagePreprocessing()
		img_prep.add_featurewise_zero_center()
		img_prep.add_featurewise_stdnorm()
		img_aug = ImageAugmentation()
		img_aug.add_random_flip_leftright()
		img_aug.add_random_rotation(max_angle=25.)
		img_aug.add_random_blur(sigma_max=3.)

		network = input_data(shape=[None, image_size, image_size, 3],
				     data_preprocessing=img_prep,
				     data_augmentation=img_aug)
		network = conv_2d(network, 32, 3, activation='relu')
		network = max_pool_2d(network, 2)
		network = conv_2d(network, 64, 3, activation='relu')
		network = conv_2d(network, 64, 3, activation='relu')
		network = max_pool_2d(network, 2)
		network = fully_connected(network, 512, activation='relu')
		network = dropout(network, 0.5)
		network = fully_connected(network, 4, activation='softmax')
		acc = Accuracy(name="Accuracy")
		network = regression(network, optimizer='adam',
				     loss='categorical_crossentropy',
				     learning_rate=0.0005, metric=acc)

		#self.model = tflearn.DNN(network, checkpoint_path='AllData.tflearn', max_checkpoints = 3,
				    #tensorboard_verbose = 3, tensorboard_dir='tmp/tflearn_logs/')
		self.model = tflearn.DNN(network)
		self.model.load("Candidates/AllData_final.tflearn_Nov25")
		

	def predict_rotation(self, img):
		
		#scaled_img = scipy.misc.imresize(img, (self.image_size, self.image_size), interp="bicubic").astype(np.float32, casting='unsafe')
		#prediction = self.model.predict([scaled_img])
		#rotation  = np.argmax(prediction[0]) 
		#print (prediction)
		#return (rotation * 90)
		return (90)


	def rotate_image(self, img, rotation, image_path):
		
		img = scipy.misc.imrotate(img, rotation, interp="bicubic")
		scipy.misc.imsave(image_path, img)

	def download_image(self, url, download_path):

		try:
			buf = urllib.urlopen(url)
		except:
			return False
		if buf.headers.maintype == 'image':
			image = file(download_path, 'wb')
			image.write(urllib.urlopen(url).read())
			image.close()
		return True

	def load_image(self, image_path):

		try:
			return scipy.ndimage.imread(image_path, mode="RGB")
		except:
			return False
	
	def full_demo(self, url, rotation):

		filename = url.split('/')[-1].split('#')[0].split('?')[0] # get filename and remove queries
		#if not (filename.lower().endswith(('.png', '.jpg', '.jpeg'))):
			#filename += imghdr.what(download_path)
		download_path = "static/" + filename 
		#rotated_path = "static/rotated_" + filename 
		#derotated_path = "static/derotated_" + filename 

		if (self.download_image(url, download_path)):
			if not (filename.lower().endswith(('.png', '.jpg', '.jpeg'))):
				filename += "." + imghdr.what(download_path)

			rotated_path = "static/rotated_" + filename 
			derotated_path = "static/derotated_" + filename 
			self.rotate_image(self.load_image(download_path), rotation, rotated_path)
			derotation = self.better_prediction(self.load_image(rotated_path))
			print (derotation)
			

			#De-rotate image
			self.rotate_image(self.load_image(rotated_path), 360 - derotation, derotated_path)
		else:
			print ("Could not download image")

		if (abs(rotation + 360 - derotation) % 360) == 0:
			print ("I am correct")
			with open('LIFETIME_SCORE', 'a') as f:
				f.write("1\n")
			
			#SESSION_SCORE will be blanked each time the server starts
			with open('SESSION_SCORE', 'a') as f: 
				f.write("1\n")
		else:
			print ("I am incorrect")
			with open('LIFETIME_SCORE', 'a') as f:
				f.write("0\n")
			#SESSION_SCORE will be blanked each time the server starts
			with open('SESSION_SCORE', 'a') as f: 
				f.write("0\n")

	def better_prediction(self, img):


		prediction = [[] for i in range(4)]
		max_val = 0
		row = 0
		for i in range(4):
			test_img = scipy.misc.imrotate(img, (i*90), interp="bicubic")
			test_img = scipy.misc.imresize(test_img, (self.image_size, self.image_size), interp="bicubic").astype(np.float32, casting='unsafe')
			prediction[i] = self.model.predict([test_img])
			#print(max(prediction[i][0]))
			if (max(prediction[i][0]) > max_val):
				row = i
				max_val = max(prediction[i][0])
		
		pprint.pprint (prediction)
		col = np.argmax(prediction[row][0])
		
		#print (prediction[row])
		#print (np.argmax(prediction[i]))
		# Check the result.
		#rotation  = np.argmax(prediction[0]) 

		#print ("Rotated by %d degrees" % ((4 + col - row)*90))
		return (90*(4 + col - row))

	def __del__(self):
		del self.model
		#self.model = None
		print ("Deleting Image_Rotater object")
