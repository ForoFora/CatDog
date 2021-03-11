import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3

from keras import backend as K

inc_model=InceptionV3(include_top=False, 
                      weights='imagenet', 
                      input_shape=((150, 150, 3)))

bottleneck_datagen = ImageDataGenerator(rescale=1./255)
    
train_generator = bottleneck_datagen.flow_from_directory('data/train/',
                                        target_size=(150, 150),
                                        batch_size=32,
                                        class_mode=None,
                                        shuffle=False)

validation_generator = bottleneck_datagen.flow_from_directory('data/validation/',
                                                               target_size=(150, 150),
                                                               batch_size=32,
                                                               class_mode=None,
                                                               shuffle=False)

bottleneck_features_train = inc_model.predict_generator(train_generator, 2000)
np.save(open('bottleneck_features/bn_features_train.npy', 'wb'), bottleneck_features_train)

bottleneck_features_validation = inc_model.predict_generator(validation_generator, 2000)
np.save(open('bottleneck_features/bn_features_validation.npy', 'wb'), bottleneck_features_validation)
