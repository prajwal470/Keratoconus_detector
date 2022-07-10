## importing the required libraries ##

import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import PreProcessing


def classify(image):
    img_array = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (100, 100))

    ## get the data set from saved hdf file ##

    with h5py.File('data.h5', 'r') as hdf:
        x_train = np.array(hdf.get('X_train'))
        y_train = np.array(hdf.get('Y_train'))

        # model = tf.keras.models.load_model("classifier.model")
        # predictions = model.predict([X])
        # # print(predictions)
        # for i in predictions[:10]:
        #     print(np.argmax(i))
        # print(Y_train[:10])

        for x, y in zip(x_train, y_train):
            if cv2.countNonZero(cv2.subtract(new_array, x)) == 0:
                if y == 1:
                    return 'infected'
                else:
                    return 'normal'


status = classify('OCT images/Infected/image_a1.png')
print(status)
PreProcessing.features('OCT images/Infected/image_a1.png')