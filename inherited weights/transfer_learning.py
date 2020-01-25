"""
Created on Fri Jan 17 19:32:57 2020

@author: Obi Ebuka David

NOTE: This code is leverages on pre-trained weights from imagenet
Transfer learning is done here

ensure you check the difference btw fit and fit_generator
"""
#import required libraries
import numpy as np
from pathlib import Path
import joblib
from keras.preprocessing import image
from keras.applications import vgg16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.models import model_from_json
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

# Path to folders with training data
dog_path = Path("training_data") / "dogs"
not_dog_path = Path("training_data") / "not_dogs"
images = []
labels = []

# Input the images to be predicted
image_to_predict_src = image.load_img("images/unknown3.jpg", target_size=(64, 64))


def import_vgg_model():
    model = vgg16.VGG16(weights='imagenet',
                        include_top=False,
                        input_shape=(64, 64, 3))
    return model


def load_images_to_array():
    # Load all the not-dog images
    for img in not_dog_path.glob("*.png"):
        # Load the images from disk
        img = image.load_img(img)
        # Convert the images to a numpy array
        image_array = image.img_to_array(img)
        # Add the images to the list of images
        images.append(image_array)
        # For each 'not dog' images, the expected value should be 0
        labels.append(0)

    # Load all the dog images
    for img in dog_path.glob("*.png"):
        # Load the images from disk
        img = image.load_img(img)
        # Convert the images to a numpy array
        image_array = image.img_to_array(img)
        # Add the images to the list of images
        images.append(image_array)
        # For each 'dog' images, the expected value should be 1
        labels.append(1)


def feature_extraction_using_pre_trained_cnn():
    load_images_to_array()
    # Create a single numpy array with all the images we loaded
    x_train = np.array(images)
    # Also convert the labels to a numpy array
    y_train = np.array(labels)
    # Normalize images data to 0-to-1 range
    x_train = vgg16.preprocess_input(x_train)
    modelvgg = import_vgg_model()
    # Extract features for each images (all in one pass)
    features_x = modelvgg.predict(x_train)
    # print extracted features
    print(features_x)
    # Save the array of extracted features to a file
    joblib.dump(features_x, "x_train.dat")

    # Save the matching array of expected values to a file
    joblib.dump(y_train, "y_train.dat")


def create_your_own_model():
    # Load data set of saved features
    x_train = joblib.load("x_train.dat")
    y_train = joblib.load("y_train.dat")

    # Create a model and add layers
    model = Sequential()
    model.add(Flatten(input_shape=x_train.shape[1:]))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=['accuracy']
    )

    # Train the model
    model.fit(
        x_train,
        y_train,
        epochs=25,
        shuffle=True
    )

    # Save neural network structure
    model_structure = model.to_json()
    f = Path("my_model_structure.json")
    f.write_text(model_structure)

    # Save neural network's trained weights
    model.save_weights("my_model_weights.h5")


def use_saved_model_structures_and_weights(image_to_predict_src):
    feature_extraction_using_pre_trained_cnn() # A call to user defined models
    create_your_own_model() # A call to user defined models
    # Load the json file that contains the model's structure
    f = Path("my_model_structure.json")
    model_structure = f.read_text()
    # Recreate the Keras model object from the json data
    model = model_from_json(model_structure)
    # Re-load the model's trained weights
    model.load_weights("my_model_weights.h5")

    # Convert the images to a numpy array
    image_array = image.img_to_array(image_to_predict_src)

    # Add a forth dimension to the images (since Keras expects a bunch of images, not a single images)
    images = np.expand_dims(image_array, axis=0)

    # Normalize the data
    images = vgg16.preprocess_input(images)

    # Use the pre-trained neural network to extract features from our test images (the same way we did to train the model)
    feature_extraction_model = import_vgg_model()
    features = feature_extraction_model.predict(images)

    return features, model


def predict_and_display_result():
    function_response = use_saved_model_structures_and_weights(image_to_predict_src)
    features, model = function_response
    results = model.predict(features)
    # Since we are only testing one images with possible class, we only need to check the first result's first element
    single_result = results[0][0]

    # Print the result
    print("Likelihood that this images contains a dog: {}%".format(int(single_result * 100)))

# A call to do it all and return results
predict_and_display_result()