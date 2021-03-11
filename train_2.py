import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout

from keras import backend as K

train_data = np.load(open('bottleneck_features/bn_features_train.npy', 'rb'))
train_labels = np.array([0] * 1000 + [1] * 1000) 

validation_data = np.load(open('bottleneck_features/bn_features_validation.npy', 'rb'))
validation_labels = np.array([0] * 1000 + [1] * 1000)

fc_model = Sequential()
fc_model.add(Flatten(input_shape=train_data.shape[1:]))
fc_model.add(Dense(64, activation='relu', name='dense_one'))
fc_model.add(Dropout(0.5, name='dropout_one'))
fc_model.add(Dense(64, activation='relu', name='dense_two'))
fc_model.add(Dropout(0.5, name='dropout_two'))
fc_model.add(Dense(1, activation='sigmoid', name='output'))

fc_model.compile(optimizer='rmsprop', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

fc_model.fit(train_data, train_labels,
            epochs=50, batch_size=32,
            validation_data=(validation_data, validation_labels))

fc_model.save_weights('bottleneck_features/fc_inception_cats_dogs_250.hdf5')

fc_model.evaluate(validation_data, validation_labels)