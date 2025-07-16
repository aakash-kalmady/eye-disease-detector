import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify

# Initialize the Flask application
app = Flask("eye-disease-detector")

# Machine Learning Model Configuration
# Path to the trained model
MODEL_PATH = 'eye_disease_model.h5' 
# Image size that must match the training size
IMG_SIZE = 128  
# The names of your classes
CATEGORIES = ['normal', 'cataract', 'diabetic_retinopathy', 'glaucoma']

# Load the Machine Learning Model
# Done once when the server starts for efficiency
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Machine learning model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading machine learning model: {e}")
    model = None

# Flask RESTful API Routes
@app.route('/', methods=['GET'])
def index():
    # Renders the main upload page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    return("Hello world")

# Run the app
if __name__ == '__main__':
    # Use port 8080 to avoid conflicts with other services
    app.run(host='0.0.0.0', port=8080, debug=True)