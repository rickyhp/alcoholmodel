# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 23:48:15 2018

@author: rp
"""

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

IMG_WIDTH = 224 # LeNet 28, VGG16 224
IMG_HEIGHT = 224 # LeNet 28, VGG16 224

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
orig = image.copy()
 
# pre-process the image for classification
image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])
 
# classify the input image
(gambling, alcohol) = model.predict(image)[0]
print("Alcohol/Gambling: ",alcohol, " ", gambling)

# build the label
label = "Alcohol" if alcohol > gambling else "Gambling"
proba = alcohol if alcohol > gambling else gambling
label = "{}: {:.2f}%".format(label, proba * 100)
 
# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)
 
# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)