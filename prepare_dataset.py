# REFER Conv_mode_train.ipynb #

import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
import os
import random

datadir = 'D:/Working Directory/PycharmProjects/KeratoconusDetector/OCT images'  #<data directory>
categories = ['Normal', 'Infected']
training_data = []


def create_train_data(IMG_SIZE):
    for category in categories:
        path = os.path.join(datadir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])


def create(IMG_SIZE=25, Test=False):
    create_train_data(IMG_SIZE)
    random.shuffle(training_data)
    if Test == True:
        for sample in training_data:
            print(sample[1])
            # changing the contrast and brightness of the image #
            # sample[0] = cv2.addWeighted(sample[0],1.8,np.zeros(sample[0].shape, sample[0].dtype),0,-120)
            plt.imshow(sample[0], cmap="gray")
            plt.show()
    X = []
    Y = []

    for feartures, label in training_data:
        Y.append(label)
        X.append(feartures)

    # X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    def save_data():
        with h5py.File('data.h5', 'w') as hdf:
            hdf.create_dataset('X_train', data=X)
            hdf.create_dataset('Y_train', data=Y)

    save_data()

    ## checking the saved data set ##
    with h5py.File('data.h5', 'r') as hdf:
        ls = list(hdf.keys())
        print(ls)
        X_train = hdf.get('X_train')
        Y_train = hdf.get('Y_train')
        print(X_train.shape)
        print(Y_train.shape)


create(100, False)

