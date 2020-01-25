"""
Created on Fri Jan 17 19:32:57 2020

@author: Obi Ebuka David

NOTE: This code learns from the provided dataset. Enjoy

ensure you check the difference btw fit and fit_generator
"""

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

image_to_predict_src = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))


def load_images():
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size=(64, 64),
                                                     batch_size=32,
                                                     class_mode='binary')

    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

    return train_datagen, test_datagen, training_set, test_set


def your_cnn_model():
    # Initialising the CNN
    model = Sequential()

    # Convolution Layer one
    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding a second convolutional layer
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattening
    model.add(Flatten())

    # Full connection
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compiling the CNN
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # load image function call
    train_datagen, test_datagen, training_set, test_set = load_images()

    model.fit_generator(training_set, steps_per_epoch=8000, epochs=25, validation_data=test_set, validation_steps=2000)

    return model


def load_image_and_predict(image_to_predict_src):
    prediction = ""
    train_datagen, test_datagen, training_set, test_set=load_images()
    image_to_predict_src_toarray = image.img_to_array(image_to_predict_src)
    image_to_predict_src_toarray = np.expand_dims(image_to_predict_src_toarray, axis=0)
    model = your_cnn_model()
    result = model.predict(image_to_predict_src_toarray)
    training_set.class_indices
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'

    return prediction


# A call to function do it all
load_image_and_predict(image_to_predict_src)

