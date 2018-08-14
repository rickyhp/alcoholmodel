# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 16:23:51 2018

@author: alphaX
"""
import os
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split


folder = 'F:\\CODE\Alcohol\\dataset\\train\\'
class_1 = 'alcohol'
class_2 = 'gambling'

data_dir = os.path.join('.', folder)
class_1_dir = os.path.join(data_dir, class_1)
class_2_dir = os.path.join(data_dir, class_2)

class_1_files = [os.path.join(class_1_dir, f) for f in os.listdir(class_1_dir)]
class_2_files = [os.path.join(class_2_dir, f) for f in os.listdir(class_2_dir)]

def process_raw_image(image):
    original = load_img(image, target_size=(224, 224))
    return img_to_array(original)

class_1_lst = list()
for image in class_1_files:
    class_1_lst.append(process_raw_image(image))
    
class_2_lst = list()
for image in class_2_files:
    class_2_lst.append(process_raw_image(image))
    
class_1_batch = np.array(class_1_lst)
class_2_batch = np.array(class_2_lst)

#create Y for class_1 and class_2
class_1_y = np.ones(class_1_batch.shape[0]) 
class_2_y = np.zeros(class_2_batch.shape[0])

#merge 2 different classes into 1 data X, y
X = np.concatenate((class_1_batch, class_2_batch))
y = np.concatenate((class_1_y, class_2_y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
data = [X_train, X_test, y_train, y_test]
with open("alcohol_gambling_data.pickle", "wb") as file:
    pickle.dump(data, file)

#image = os.path.join(class_1_dir, '20. cmh8pphqgiej8tbbvuqnntrq5umqhqlf_598x414.jpg')
#
## load an image in PIL format
#original = load_img(image, target_size=(224, 224))
#plt.imshow(original)
#plt.show()
#print('PIL image size', original.size)
#
## convert the PIL image to a numpy array
## IN PIL - image is in (width, height, channel)
## In Numpy - image is in (height, width, channel)
#numpy_image = img_to_array(original)
#plt.imshow(np.uint8(numpy_image))
#plt.show()
#print('numpy array size',numpy_image.shape)
#
## Convert the image / images into batch format
## expand_dims will add an extra dimension to the data at a particular axis
## We want the input matrix to the network to be of the form (batchsize, height, width, channels)
## Thus we add the extra dimension to the axis 0.
#image_batch = np.expand_dims(numpy_image, axis=0)
#plt.imshow(np.uint8(image_batch[0]))
#print('image batch size', image_batch.shape)
