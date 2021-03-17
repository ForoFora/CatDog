import numpy as np
import tensorflow as tf
from tensorflow import keras

from flask import Flask, jsonify, request
 
app = Flask(__name__)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D

from keras import backend as K

# dimensions of our images.
img_width, img_height = 150, 150

batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.load_weights('./first_try.h5')

@app.route("/")
def hello():
    return "Hello, World!"

@app.route('/api/v1.0/img', methods=['POST'])
def get_tasks():
    req_data = request.get_json()
    np_x = np.array(req_data["img"]).reshape(input_shape)
    global model
    prediction = model.predict(x=np.array([np_x / 255.]))
    print(prediction)
    prediction = prediction + 0.00001
    return str(prediction)

if __name__ == "__main__":
    app.run()