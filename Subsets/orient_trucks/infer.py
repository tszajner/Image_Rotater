  # -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

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

parser = argparse.ArgumentParser(description='Decide if an image is a picture of a plane')
parser.add_argument('image', type=str, help='The image image file to check')
args = parser.parse_args()


# Same network definition as before
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

network = input_data(shape=[None, 64, 64, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
acc = Accuracy(name="Accuracy")
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0005, metric=acc)

model = tflearn.DNN(network, checkpoint_path='model_truck_6.tflearn', max_checkpoints = 3,
                    tensorboard_verbose = 3, tensorboard_dir='tmp/tflearn_logs/')
model.load("model_truck_6.tflearn-36")

# Load the image file
img = scipy.ndimage.imread(args.image, mode="RGB")
img_180 = scipy.misc.imrotate(img, 90, interp="bicubic")

# Scale it to 32x32
img = scipy.misc.imresize(img, (64, 64), interp="bicubic").astype(np.float32, casting='unsafe')
img_180 = scipy.misc.imresize(img_180, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')

# Predict
prediction = model.predict([img])
print(prediction)

# Check the result.
is_bird = np.argmax(prediction[0]) == 1

if is_bird:
    print("Thats a dog!")
else:
    print("Thats a cat!")
