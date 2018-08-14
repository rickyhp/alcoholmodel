# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 15:24:20 2018

@author: alfredt
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import pickle
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

with open("alcohol_gambling_splitted.pickle", "rb") as handle:
    X_train, X_test, y_train, y_test = pickle.load(handle)
    
#split test data into dev and test set
X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

#normalize the data
X_train_norm = X_train / 255
X_dev_norm = X_dev / 255
X_test_norm = X_test / 255

input_shape = X_train_norm.shape[1:]

vgg16_model = VGG16(weights='imagenet', include_top=False)
in_layer = Input(shape=input_shape, name='image_input')
x = Flatten()(vgg16_model(in_layer))
x = Dense(2056, activation='relu')(x)
x = Dropout(rate=0.5)(x)
x = Dense(1028, activation='relu')(x)
x = Dropout(rate=0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(rate=0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(rate=0.5)(x)
prediction = Dense(2, activation='sigmoid')(x)

model = Model(inputs=in_layer, outputs=prediction)

for layer in model.layers[:-9]:
    print(layer)
    layer.trainable = False
    
print("Trainable")
for layer in model.layers[-9:]:
    print(layer)
    layer.trainable = True
    
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, decay=1e-4),metrics=['accuracy'])

#set up VGG16 Model
batch_size = 32 
epochs = 10

model.fit(X_train_norm, y_train, validation_data=(X_dev_norm, y_dev), epochs=epochs, batch_size=batch_size, verbose=1)
score = model.evaluate(X_test_norm, y_test, batch_size=batch_size)
print("Testing Accuracy: %.4f%% | Loss: %.4f" % (score[1], score[0]))

score = model.evaluate(X_dev_norm, y_dev, batch_size=batch_size)
print("Validation Accuracy: %.4f%% | Loss: %.4f" % (score[1], score[0]))

# save the model to disk
print("[INFO] serializing network...")
model.save("vgg_alcohol_gambling_v2.model")

# serialize model to JSON
#model_json = model.to_json()
#with open("vgg_alcohol_gambling_v2.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("vgg_alcohol_gambling_v2.h5")
#print("Saved model to disk")