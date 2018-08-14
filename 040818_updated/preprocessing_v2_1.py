# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 14:05:31 2018

@author: alfredt
"""

import os
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split

folder = 'images'
class_1 = '1_Gambling_Game_Type'

data_dir = os.path.join('.', folder)
class_1_dir = os.path.join(data_dir, class_1)

class_1_files = [os.path.join(class_1_dir, f) for f in os.listdir(class_1_dir)]

def process_raw_image(image):
    original = load_img(image, target_size=(224, 224))
    return img_to_array(original)

class_1_lst = list()
for image in class_1_files:
    class_1_lst.append(process_raw_image(image))
    
class_1_batch = np.array(class_1_lst)

#create Y for class_1 and class_2
class_1_y = np.ones(class_1_batch.shape[0])

data = [class_1_batch, class_1_y]
with open("class_1.pickle", "wb") as file:
    pickle.dump(data, file)