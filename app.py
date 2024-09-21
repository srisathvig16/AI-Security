import os
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Define path for uploading images
UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained model
model = load_model('mask_detector_model.h5')  # Ensure this path is correct

# Define image size to match the input size of the model
IMAGE_SIZE = 224

# Define function to make prediction
def predict_mask(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize image

    # Make prediction
    prediction = model.predict(img)
    mask, no_mask = prediction[0]

    if mask > no_mask:
        return "threat"  # Mask is detected
    else:
        return "safe"  # No mask detected

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Predict result
        result = predict_mask(file_path)

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)