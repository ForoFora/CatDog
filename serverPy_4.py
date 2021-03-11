import numpy as np
import tensorflow as tf
from tensorflow import keras

from flask import Flask, jsonify, request
app = Flask(__name__)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.layers import Dense, Flatten, Dropout

from keras import backend as K

weights_filename='new_model_weights/weights-improvement-02-0.88.hdf5'

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

@app.route("/")
def hello():
    return "Hello, World!"

@app.route('/api/v1.0/img', methods=['POST'])
def get_tasks():
    req_data = request.get_json()
    np_x = np.array(req_data["img"]).reshape(150, 150, 3)
    global model
    prediction = model.predict(x=np.array([np_x / 255.]))
    print(prediction)
    prediction = prediction + 0.0000012
    return str(prediction)

if __name__ == "__main__":
    app.run()