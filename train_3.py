import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.layers import Dense, Flatten, Dropout

from keras import backend as K

weights_filename='bottleneck_features/fc_inception_cats_dogs_250.hdf5'

inc_model=InceptionV3(include_top=False, 
                      weights='imagenet', 
                      input_shape=((150, 150, 3)))

x = Flatten()(inc_model.output)
x = Dense(64, activation='relu', name='dense_one')(x)
x = Dropout(0.5, name='dropout_one')(x)
x = Dense(64, activation='relu', name='dense_two')(x)
x = Dropout(0.5, name='dropout_two')(x)
top_model=Dense(1, activation='sigmoid', name='output')(x)
model = Model(inputs=inc_model.input, outputs=top_model)

model.load_weights(weights_filename, by_name=True)

for layer in inc_model.layers[:205]:
    layer.trainable = False

model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

filepath="new_model_weights/weights-improvement-{epoch:02d}-{accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train/',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation/',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit(
  train_generator,
  steps_per_epoch=62,
  epochs=3, 
  callbacks=callbacks_list,
  validation_data=validation_generator, 
  validation_steps=62)