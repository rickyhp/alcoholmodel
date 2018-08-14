from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, Flatten, Dropout
from keras import backend as K
from keras.models import Model

class VGG16_model:
    @staticmethod
    def build(inputShape):
        vgg16_model = VGG16(weights='imagenet', include_top=False)
        in_layer = Input(shape=inputShape, name='image_input')
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
        
        return model