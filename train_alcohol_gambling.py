# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import tensorflow as tf

tf.test.gpu_device_name()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 20
INIT_LR = 1e-3
BS = 32
IMG_WIDTH = 224 # LeNet 28, VGG16 224
IMG_HEIGHT = 224 # LeNet 28, VGG16 224
NUM_OF_CLASSES = 3
 
# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []
 
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
print(imagePaths)
random.seed(42)
random.shuffle(imagePaths)

i = 1
# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
   print(i,'.',imagePath)
   image = cv2.imread(imagePath)
   try:   
       image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
       image = img_to_array(image)
       data.append(image)
   except:
       pass
   
	# extract the class label from the image path and update the
	# labels list
   label = imagePath.split(os.path.sep)[-2]
   label = 0 if label == "alcohol" else 1 if label == "gambling" else 2
   labels.append(label)
   i = i + 1
   #print("loaded : ",i)
	
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
 
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

print('len(labels):',len(labels))
print('len(trainY):',len(trainY))
print('len(testY):',len(testY))

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=NUM_OF_CLASSES)
testY = to_categorical(testY, num_classes=NUM_OF_CLASSES)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
	
# initialize the model
print("[INFO] compiling model...")

# LE NET
from lenet import LeNet
model = LeNet.build(width=IMG_WIDTH, height=IMG_HEIGHT, depth=3, classes=NUM_OF_CLASSES)

#from vgg16 import VGG16_model
#model = VGG16_model.build(width=IMG_WIDTH, height=IMG_HEIGHT, depth=3, classes=NUM_OF_CLASSES)

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)
 
# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Alcohol/Gambling")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

#Confution Matrix and Classification Report
test_dir = 'F:\\CODE\Alcohol\\dataset\\validation'

test_generator = aug.flow_from_directory(
        test_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        color_mode="rgb",
        shuffle = False,
        class_mode='categorical',
        batch_size=1)
filenames = test_generator.filenames
nb_samples = len(filenames)

Y_pred = model.predict_generator(test_generator, steps = nb_samples)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))