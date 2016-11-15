"""
Based on the tflearn CIFAR-10 example at:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
"""

from __future__ import division, print_function, absolute_import

from skimage import color, io
from scipy.misc import imresize, imrotate
import numpy as np
from sklearn.cross_validation import train_test_split
import os
from glob import glob

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy

###################################
### Import picture files 
###################################

files_path = '/root/sharedfolder/ImageRotater/rawdata/'


files_path = os.path.join(files_path, 'lfw/*/*.jpg')
files = sorted(glob(files_path))

n_files = len(files)
print(n_files)

size_image = 32

allX = np.zeros((n_files*4, size_image, size_image, 3), dtype='float64')
ally = np.zeros(n_files*4)
count = 0
for f in files:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[count] = 0
        count += 1

	angle = 1
	upside_downimage = imrotate(new_img, angle*90)
        allX[count] = np.array(upside_downimage)
        ally[count] = angle
        count += 1

	angle = 2
	upside_downimage = imrotate(new_img, angle*90)
        allX[count] = np.array(upside_downimage)
        ally[count] = angle
        count += 1

	angle = 3
	upside_downimage = imrotate(new_img, angle*90)
        allX[count] = np.array(upside_downimage)
        ally[count] = angle
        count += 1
    except:
        continue

###################################
# Prepare train & test samples
###################################

# test-train split   
X, X_test, Y, Y_test = train_test_split(allX, ally, test_size=0.1, random_state=42)
#X, Y = shuffle(X, Y)

# encode the Ys
Y = to_categorical(Y, 4)
Y_test = to_categorical(Y_test, 4)


###################################
# Image transformations
###################################

# normalisation of images
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping & rotating images
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

###################################
# Define network architecture
###################################

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, size_image, size_image, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# 1: Convolution layer with 32 filters, each 3x3x3
network = conv_2d(network, 32, 3, activation='relu')

# 2: Max pooling layer
network = max_pool_2d(network, 2)

# 3: Convolution layer with 64 filters
network = conv_2d(network, 64, 3, activation='relu')

# 4: Convolution layer with 64 filters
network = conv_2d(network, 64, 3, activation='relu')

# 5: Max pooling layer
network = max_pool_2d(network, 2)

# 6: Fully-connected 512 node layer
network = fully_connected(network, 512, activation='relu')

# 7: Dropout layer to combat overfitting
network = dropout(network, 0.5)

# 8: Fully-connected layer with four outputs
network = fully_connected(network, 4, activation='softmax')

# Configure how the network will be trained
acc = Accuracy(name="Accuracy")
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0005, metric=acc)

# Wrap the network in a model object
model = tflearn.DNN(network, checkpoint_path='face.tflearn', max_checkpoints = 3,
                    tensorboard_verbose = 3, tensorboard_dir='tmp/tflearn_logs/')

###################################
# Train model for 100 epochs
###################################
model.fit(X, Y, validation_set=(X_test, Y_test), batch_size=500,
      n_epoch=100, run_id='face', show_metric=True)

model.save('face_final.tflearn')


