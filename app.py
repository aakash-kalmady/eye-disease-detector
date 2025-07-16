import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify

# Initialize the Flask application
app = Flask("eye-disease-detector")

# --- Flask REST API Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    return("Hello world")

# --- Run the App ---
if __name__ == '__main__':
    # Use port 8080 to avoid conflicts with other services
    app.run(host='0.0.0.0', port=8080, debug=True)