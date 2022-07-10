# importing the required libraries

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import h5py
import numpy as np
import matplotlib.pyplot as plt

# get the data set from saved hdf file

with h5py.File('data.h5', 'r') as hdf:
    X_train = np.array(hdf.get('X_train'))
    Y_train = np.array(hdf.get('Y_train'))
    print(Y_train.shape)

# normalizing the data set

X = tf.keras.utils.normalize(X_train, axis=1)


# Convolution Neural Network

def trainmodel():
    model = Sequential()
    model.add(Conv2D(128, (7, 7), input_shape=X.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (7, 7)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64, activation=tf.nn.relu))

    model.add(Dense(32, activation=tf.nn.relu))

    model.add(Dense(2, activation=tf.nn.softmax))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

    model.fit(X, Y_train, epochs=5, validation_split=0.1)

    # saving the model

    model.save('classifier.model')

    # predictions = model.predict([X])
    # print(predictions)
    # print(np.argmax(predictions[0:10]))

trainmodel()

# loading the model

saved_model = tf.keras.models.load_model("classifier.model")
predictions = saved_model.predict([X])
# print(predictions)
for i in predictions[:10]:
    print(np.argmax(i))
print(Y_train[:10])
