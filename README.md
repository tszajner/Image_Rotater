# Image Rotater

A simple convolutional neural network to determine if an image is rotated by (0, 90, 180, 270) degrees. Works most™ of the time

##Usage:

`python demo_test.py <image_url> <rotation_angle>`

saves the original, rotated, and derotated image in /static

OR

`python demo.py`

starts a webserver at localhost:6006

##TODO:
- [ ] use a better neural network architecture
- [ ] make sure demo is always able to download the image
- [ ] make sure the download code in the web server and demo code is the same source
- [ ] add a usage/help message for demo_test.py
- [ ] find a way to load in more data in train.py
- [ ] cleanup training script (selected_images)
- [ ] provide downloadable link for final model and zipped rawdata (maybe pickled)
- [ ] maybe make a python script for downloading and preprocessing all data
- [ ] cleanup Image_Rotater and provide debug info/option
- [ ] remove the subprocess call to a script and do the demo correctly
- [ ] make sure all variables have good names
