"""
Created on Fri Jan 17 19:32:57 2020

@author: Obi Ebuka David

NOTE: This code is leverages on pre-trained weights from imagenet
Transfer learning is not done here
"""
#import required libraries
import numpy as np
from keras.preprocessing import image
from keras.applications import vgg16

#input the images to be predicted
image_to_predict_src=image.load_img("images/unknown1.jpg", target_size=(224, 224))


def import_vgg_model():
    model = vgg16.VGG16(include_top=True,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=None,
                        pooling=None,
                        classes=1000)
    return model


def image_to_predict(image_to_predict_src):
    # Convert the images to a numpy array
    processed_image = image.img_to_array(image_to_predict_src)

    # Add a fourth dimension (since Keras expects a list of images)
    processed_image = np.expand_dims(processed_image, axis=0)

    # Normalize the input images's pixel values to the range used when training the neural network
    processed_image = vgg16.preprocess_input(processed_image)

    return processed_image


def process_and_predict():
    # Run the images through the deep neural network to make a prediction
    processed_image = image_to_predict(image_to_predict_src)
    model = import_vgg_model()
    predictions = model.predict(processed_image)
    # Look up the names of the predicted classes. Index zero is the results for the first images.
    predicted_classes = vgg16.decode_predictions(predictions)
    print("Top predictions for this images:")

    for imagenet_id, name, likelihood in predicted_classes[0]:
        print("Prediction: {} - {:2f}".format(name, likelihood))


#Run it all to view predicted result
process_and_predict()