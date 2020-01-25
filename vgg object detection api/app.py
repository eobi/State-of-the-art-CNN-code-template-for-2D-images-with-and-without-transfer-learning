"""
Created on Fri Jan 17 19:32:57 2020

@author: Obi Ebuka David

NOTE: if your app crashes after the first prediction, dont worry you are not alone, would upload a fix for you
you can check out other people having same issue and the root cause of the issue here:
https://stackoverflow.com/questions/52353540/trying-to-wrap-up-a-keras-model-in-a-flask-rest-app-but-getting-a-valueerror

"""
import numpy as np
from keras.preprocessing import image
from keras.applications import vgg16
from flask import json
from flask import Flask, request
from werkzeug.utils import secure_filename
from keras import backend

app = Flask(__name__)


@app.route('/')
def hello_world():
    return "Hello World, this is a simple flask api snippet"


@app.route('/api/upload', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
       f = request.files['file']
       f.save(secure_filename(f.filename))
       res = vgg_model_predict(f.filename)

       namelist=[]
       likelihoodlist = []

       # Top predictions for this image return as json response
       for imagenet_id, name, likelihood in res:
           namelist.append(name)
           likelihoodlist.append(str(likelihood))

       data = { "likelihood": likelihoodlist, "names": namelist}
       response = app.response_class(
          response=json.dumps(data),
          status=200,
          mimetype='application/json'
       )
       return response


def vgg_model_predict(src):
    model = vgg16.VGG16()
    # Load the image file, resizing it to 224x224 pixels (required by this model)
    img = image.load_img(src, target_size=(224, 224))
    # Convert the image to a numpy array
    x = image.img_to_array(img)
    # Add a fourth dimension (since Keras expects a list of images)
    x = np.expand_dims(x, axis=0)
    # Normalize the input image's pixel values to the range used when training the neural network
    x = vgg16.preprocess_input(x)
    # Run the image through the deep neural network to make a prediction
    predictions = model.predict(x)
    # Look up the names of the predicted classes. Index zero is the results for the first image.
    predicted_classes = vgg16.decode_predictions(predictions)

    return predicted_classes[0]




if __name__ == '__main__':
    app.run()
