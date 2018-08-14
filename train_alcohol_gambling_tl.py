# We use VGG16 as feature extractor and train a new model
# based on it
# https://www.learnopencv.com/keras-tutorial-transfer-learning-using-pre-trained-models/

import tensorflow as tf

tf.test.gpu_device_name()

from keras.applications import VGG16
#from keras.applications import resnet50
 
model = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

#model = resnet50.ResNet50(weights='imagenet',
#                                 include_top=False,
#                                 input_shape=(224, 224, 3))
				  
train_dir = 'F:\\CODE\Alcohol\\dataset\\train'
validation_dir = 'F:\\CODE\Alcohol\\dataset\\validation'
 
nTrain = 8381
nVal = 1105
NUM_OF_CLASSES=2

from keras.preprocessing.image import ImageDataGenerator
import numpy as np

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20
 
train_features = np.zeros(shape=(nTrain, 7, 7, 512))
train_labels = np.zeros(shape=(nTrain,NUM_OF_CLASSES))

validation_features = np.zeros(shape=(nVal, 7, 7, 512))
validation_labels = np.zeros(shape=(nVal,NUM_OF_CLASSES))

# Train 
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

# Validation
validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)
	
i = 0
for inputs_batch, labels_batch in train_generator:
    features_batch = model.predict(inputs_batch)
    train_features[i * batch_size : (i + 1) * batch_size] = features_batch
    train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nTrain:
        break

i = 0
for inputs_batch, labels_batch in validation_generator:
    features_batch = model.predict(inputs_batch)
    validation_features[i * batch_size : (i + 1) * batch_size] = features_batch
    validation_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nVal:
        break

train_features = np.reshape(train_features, (nTrain, 7 * 7 * 512))
validation_features = np.reshape(validation_features, (nVal, 7 * 7 * 512))

from keras import models
from keras import layers
from keras import optimizers

# Simple feedforward network with a softmax output layer
#model = models.Sequential()
#model.add(layers.Dense(256, activation='relu', input_dim=7 * 7 * 512))
#model.add(layers.Dropout(0.5))
#model.add(layers.Dense(NUM_OF_CLASSES, activation='softmax'))

# Alfred
model = models.Sequential()
model.add(layers.Dense(2056, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1028, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(NUM_OF_CLASSES, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])
 
history = model.fit(train_features,
                    train_labels,
                    epochs=20,
                    batch_size=batch_size,
                    validation_data=(validation_features,validation_labels))
					
# save the model to disk
print("[INFO] serializing network...")
model.save("alcohol_gambling_vgg16.model")