from __future__ import division, print_function, absolute_import

import scipy.ndimage
import numpy as np
import sys

import os
from glob import glob
import urllib
import time

import Image_Rotater

url = sys.argv[1]
rotation = int(sys.argv[2])
print (url)
print (rotation)


demo = Image_Rotater.Image_Rotater()
demo.full_demo(url, rotation)
