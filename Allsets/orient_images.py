"""
Based Cameron Mence's example at
http://www.subsubroutine.com/sub-subroutine/2016/9/30/cats-and-dogs-and-convolutional-neural-networks ,

and Adam Geitgey's example at
https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721#.nt77j3ty4 ,

which in turn are both based on the tflearn CIFAR-10 example at:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
"""

from __future__ import division, print_function, absolute_import

from skimage import color, io
from scipy.misc import imresize, imrotate
import numpy as np
from sklearn.cross_validation import train_test_split
import os
from glob import glob

import random 


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

rawdata_path = '/root/sharedfolder/ImageRotater/rawdata/'
selected_images = ['lfw/*/*.jpg',
		'101_ObjectCategories/bonsai/*',
		'101_ObjectCategories/brontosaurus/*',
		'101_ObjectCategories/buddha/*',
		'101_ObjectCategories/car_side/*',
		'101_ObjectCategories/chair/*',
		'101_ObjectCategories/chandelier/*',
		'101_ObjectCategories/cougar_body/*',
		'101_ObjectCategories/cougar_face/*',
		'101_ObjectCategories/cup/*',
		'101_ObjectCategories/dalmatian/*',
		'101_ObjectCategories/elephant/*',
		'101_ObjectCategories/emu/*',
		'101_ObjectCategories/ewer/*',
		'101_ObjectCategories/Faces/*',
		'101_ObjectCategories/Faces_easy/*',
		'101_ObjectCategories/ferry/*',
		'101_ObjectCategories/flamingo/*',
		'101_ObjectCategories/flamingo_head/*',
		'101_ObjectCategories/garfield/*',
		'101_ObjectCategories/gramophone/*',
		'101_ObjectCategories/grand_piano/*',
		'101_ObjectCategories/hedgehog/*', #MAYBE
		'101_ObjectCategories/helicopter/*',
		'101_ObjectCategories/ibis/*',
		'101_ObjectCategories/inline_skate/*', #MAYBE
		'101_ObjectCategories/joshua_tree/*', 
		'101_ObjectCategories/kangaroo/*', 
		'101_ObjectCategories/ketch/*', 
		'101_ObjectCategories/lamp/*', 
		'101_ObjectCategories/laptop/*', 
		'101_ObjectCategories/Leopards/*', 
		'101_ObjectCategories/llama/*', 
		'101_ObjectCategories/menorah/*', 
		'101_ObjectCategories/metronome/*', 
		'101_ObjectCategories/Motorbikes/*', 
		'101_ObjectCategories/okapi/*', 
		'101_ObjectCategories/panda/*',  #MAYBE
		'101_ObjectCategories/pigeon/*',  #MAYBE
		'101_ObjectCategories/pyramid/*', 
		'101_ObjectCategories/rhino/*', 
		'101_ObjectCategories/rooster/*', 
		'101_ObjectCategories/saxophone/*', #MAYBE
		'101_ObjectCategories/schooner/*',
		'101_ObjectCategories/sea_horse/*', #MAYBE
		'101_ObjectCategories/stegosaurus/*', 
		'101_ObjectCategories/umbrella/*', #MAYBE
		'101_ObjectCategories/wheelchair/*', 
		'101_ObjectCategories/wild_cat/*', 
		'101_ObjectCategories/windsor_chair/*', 
		'256_ObjectCategories/003.backpack/*', #Maybe
		'256_ObjectCategories/008.bathtub/*', #Maybe
		'256_ObjectCategories/009.bear/*',
		'256_ObjectCategories/010.beer-mug/*',
		'256_ObjectCategories/011.billiards/*',
		'256_ObjectCategories/013.birdbath/*',
		'256_ObjectCategories/015.bonsai-101/*',
		'256_ObjectCategories/016.boom-box/*',
		'256_ObjectCategories/018.bowling-pin/*',
		'256_ObjectCategories/025.cactus/*', #Maybe
		'256_ObjectCategories/028.camel/*',
		'256_ObjectCategories/032.cartman/*',
		'256_ObjectCategories/036.chandelier-101/*',
		'256_ObjectCategories/038.chimp/*',
		'256_ObjectCategories/041.coffee-mug/*',
		'256_ObjectCategories/046.computer-monitor/*',
		'256_ObjectCategories/049.cormorant/*',
		'256_ObjectCategories/050.covered-wagon/*',
		'256_ObjectCategories/051.cowboy-hat/*', #Maybe
		'256_ObjectCategories/053.desk-globe/*',
		'256_ObjectCategories/056.dog/*',
		'256_ObjectCategories/060.duck/*',
		'256_ObjectCategories/062.eiffel-tower/*',
		'256_ObjectCategories/064.elephant-101/*',
		'256_ObjectCategories/065.elk/*',
		'256_ObjectCategories/066.ewer-101/*',
		'256_ObjectCategories/070.fire-extinguisher/*',
		'256_ObjectCategories/071.fire-hydrant/*',
		'256_ObjectCategories/072.fire-truck/*',
		'256_ObjectCategories/076.football-helmet/*',
		'256_ObjectCategories/083.gas-pump/*',
		'256_ObjectCategories/084.giraffe/*',
		'256_ObjectCategories/085.goat/*',
		'256_ObjectCategories/086.golden-gate-bridge/*',
		'256_ObjectCategories/089.goose/*',
		'256_ObjectCategories/090.gorilla/*',
		'256_ObjectCategories/091.grand-piano-101/*',
		'256_ObjectCategories/099.harpsichord/*',
		'256_ObjectCategories/102.helicopter-101/*',
		'256_ObjectCategories/104.homer-simpson/*',
		'256_ObjectCategories/105.horse/*',
		'256_ObjectCategories/107.hot-air-balloon/*',
		'256_ObjectCategories/113.hummingbird/*',
		'256_ObjectCategories/114.ibis-101/*',
		'256_ObjectCategories/119.jesus-christ/*',
		'256_ObjectCategories/121.kangaroo-101/*',
		'256_ObjectCategories/123.ketch-101/*',
		'256_ObjectCategories/127.laptop-101/*',
		'256_ObjectCategories/129.leopards-101/*',
		'256_ObjectCategories/132.light-house/*',
		'256_ObjectCategories/134.llama-101/*',
		'256_ObjectCategories/140.menorah-101/*',
		'256_ObjectCategories/141.microscope/*',
		'256_ObjectCategories/143.minaret/*',
		'256_ObjectCategories/144.minotaur/*',
		'256_ObjectCategories/145.motorbikes-101/*',
		'256_ObjectCategories/146.mountain-bike/*',
		'256_ObjectCategories/151.ostrich/*',
		'256_ObjectCategories/152.owl/*',
		'256_ObjectCategories/154.palm-tree/*',
		'256_ObjectCategories/158.penguin/*',
		'256_ObjectCategories/159.people/*',
		'256_ObjectCategories/162.picnic-table/*',
		'256_ObjectCategories/165.pram/*',
		'256_ObjectCategories/167.pyramid/*',
		'256_ObjectCategories/168.raccoon/*',
		'256_ObjectCategories/169.radio-telescope/*',
		'256_ObjectCategories/174.rotary-phone/*',
		'256_ObjectCategories/178.school-bus/*',
		'256_ObjectCategories/181.segway/*',
		'256_ObjectCategories/182.self-propelled-lawn-mower/*',
		'256_ObjectCategories/186.skunk/*',
		'256_ObjectCategories/188.smokestack/*',
		'256_ObjectCategories/197.speed-boat/*',
		'256_ObjectCategories/205.superman/*',
		'256_ObjectCategories/207.swan/*',
		'256_ObjectCategories/212.teapot/*',
		'256_ObjectCategories/213.teddy-bear/*',
		'256_ObjectCategories/214.teepee/*',
		'256_ObjectCategories/217.tennis-court/*', #Maybe
		'256_ObjectCategories/219.theodolite/*',
		'256_ObjectCategories/220.toaster/*',
		'256_ObjectCategories/223.top-hat/*',
		'256_ObjectCategories/224.touring-bike/*',
		'256_ObjectCategories/225.tower-pisa/*',
		'256_ObjectCategories/227.treadmill/*',
		'256_ObjectCategories/228.triceratops/*',
		'256_ObjectCategories/229.tricycle/*',
		'256_ObjectCategories/231.tripod/*',
		'256_ObjectCategories/232.t-shirt/*',
		'256_ObjectCategories/235.umbrella-101/*',
		'256_ObjectCategories/236.unicorn/*',
		'256_ObjectCategories/241.waterfall/*',
		'256_ObjectCategories/244.wheelbarrow/*',
		'256_ObjectCategories/245.windmill/*',
		'256_ObjectCategories/246.wine-bottle/*',
		'256_ObjectCategories/250.zebra/*',
		'256_ObjectCategories/251.airplanes-101/*',
		'256_ObjectCategories/252.car-side-101/*',
		'256_ObjectCategories/253.faces-easy-101/*',
		'256_ObjectCategories/254.greyhound/*',
		'256_ObjectCategories/255.tennis-shoes/*', #Maybe
		'256_ObjectCategories/256.toad/*',
		'cifar10/*/automobile/*',
		'cifar10/*/bird/*',
		'cifar10/*/cat/*',
		'cifar10/*/deer/*',
		'cifar10/*/dog/*',
		'cifar10/*/horse/*',
		'cifar10/*/ship/*',
		'cifar10/*/truck/*',
		'cifar100/coarse/*/food_containers/bottle/*',
		'cifar100/coarse/*/food_containers/cup/*',
		'cifar100/coarse/*/household_electrical_devices/lamp/*',
		'cifar100/coarse/*/household_furniture/chair/*',
		'cifar100/coarse/*/household_furniture/couch/*',
		'cifar100/coarse/*/household_furniture/table/*',
		'cifar100/coarse/*/large_carnivores/*/*',
		'cifar100/coarse/*/large_man-made_outdoor_things/*/*',
		'cifar100/coarse/*/large_natural_outdoor_scenes/forest/*',
		'cifar100/coarse/*/large_natural_outdoor_scenes/mountain/*',
		'cifar100/coarse/*/large_natural_outdoor_scenes/plain/*',
		'cifar100/coarse/*/large_natural_outdoor_scenes/sea/*',
		'cifar100/coarse/*/large_omnivores_and_herbivores/*/*',
		'cifar100/coarse/*/medium_mammals/fox/*',
		'cifar100/coarse/*/medium_mammals/possum/*',
		'cifar100/coarse/*/medium_mammals/raccoon/*',
		'cifar100/coarse/*/people/boy/*',
		'cifar100/coarse/*/people/girl/*',
		'cifar100/coarse/*/people/man/*',
		'cifar100/coarse/*/people/woman/*',
		'cifar100/coarse/*/reptiles/dinosaur/*',
		'cifar100/coarse/*/small_mammals/rabbit/*',
		'cifar100/coarse/*/trees/*/*',
		'cifar100/coarse/*/vehicles_1/*/*',
		'cifar100/coarse/*/vehicles_2/streetcar/*' ,
		'cifar100/coarse/*/vehicles_2/tank/*' ,
		'cifar100/coarse/*/vehicles_2/tractor/*' ,
		'cifar100/coarse/*/vehicles_2/lawn_mower/*',
		'English/Hnd/Img/*/*',
		'English/Img/GoodImg/Bmp/*/*' ]
		
files = []

for directory in selected_images:
	files_path = os.path.join(rawdata_path, directory)
	#print (len(sorted(glob(files_path))))
	if (len(sorted(glob(files_path)))) == 0:
		print (directory)
	files += sorted(glob(files_path))
	n_files = len(files)
	#print(n_files)

random.shuffle(files)
files = files[:90000]
n_files = len(files)
print(n_files)

size_image = 32

allX = np.zeros((n_files*4, size_image, size_image, 3), dtype='float32')
ally = np.zeros(n_files*4)
count = 0
for f in files:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
	if random.choice([True, False]):
		if 'English' not in f: # letters/numbers shouldn't be flipped
			#print (f)
			new_img = np.fliplr(new_img)
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

#Lets do this earlier to prevent 90 degree images being represented as 270, and vice-versa
#img_aug.add_random_flip_leftright()

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
model = tflearn.DNN(network, checkpoint_path='AllData.tflearn', max_checkpoints = 3,
                    tensorboard_verbose = 3, tensorboard_dir='tmp/tflearn_logs/')

###################################
# Train model for 100 epochs
###################################
model.fit(X, Y, validation_set=(X_test, Y_test), batch_size=500,
      n_epoch=100, run_id='AllData', show_metric=True)

model.save('AllData_final.tflearn')



