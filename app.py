from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# all the functions and stuff from keras for our predictions
from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# filename of the weights
MODEL_WEIGHTS = 'models/weights-02.hdf5'
# number of classes
NUM_CLASSES = 2


@app.route('/')
def index():
    '''
    render index.html page
    '''
    return render_template('index.html')

def get_model():
    '''
    construct the model and return the model
    '''

    # load the MobileNet architecture
    base_model = MobileNet(input_shape=(128,128,3), alpha=0.75, weights='imagenet', include_top=False)

    # add our custom layers on top, add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    x = Dense(128, activation='relu')(x)

    # and a logistic layer -- we have NUM_CLASSES=2 (cats and dogs)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

@app.route('/predict', methods=['POST'])
def predict():
    '''
    read the image file uploaded by the user.
    put a BytesIO wrapper on it. open it using PIL.Image
    convert it into numpy array and make a prediction
    and finally render prediction.html page
    '''
    # read the image and resize it appropriately for our model
    img_file = request.files['image']
    img = img_file.read()
    img = BytesIO(img)
    img = Image.open(img)
    img = img.resize((128,128))
    img = np.array(img)
    img = np.resize(img, (1,img.shape[0], img.shape[1], img.shape[2]))
    # print(img.shape) # rows x cols x channels

    # make predictions on all images in "predict/images" folder using test_generator
    # and print the predictions one by one
    results = model.predict(img)
    results *= 100
    print(results)

    return render_template('prediction.html', results=results, img=img)
    
if __name__ == '__main__':
    # get the model architecture
    model = get_model()
    # load trained weights, these weights are trained by running model.py
    model.load_weights(MODEL_WEIGHTS)
    # run the app
    app.run(debug=False)