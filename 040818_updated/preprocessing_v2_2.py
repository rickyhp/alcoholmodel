# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 14:50:52 2018

@author: alfredt
"""
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

with open("class_1.pickle", "rb") as handle:
    class_1_batch, class_1_y = pickle.load(handle)
    
with open("class_2.pickle", "rb") as handle:
    class_2_batch, class_2_y = pickle.load(handle)
    
y_labels = np.concatenate((class_1_y, class_2_y))

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_labels)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y_onehot = onehot_encoder.fit_transform(integer_encoded)

X = np.concatenate((class_1_batch, class_2_batch))

data = [X, y_onehot]
with open("alcohol_gambling.pickle", "wb") as file:
    pickle.dump(data, file, protocol=4)
    
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.20, random_state=42, stratify=y_onehot)
data = [X_train, X_test, y_train, y_test]
with open("alcohol_gambling_splitted.pickle", "wb") as file:
    pickle.dump(data, file, protocol=4)
