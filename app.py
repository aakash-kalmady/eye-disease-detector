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

# Reads an uploaded image file and preprocesses it to the format the model expects
def preprocess_image(image_file):
    # Read the image file stream from the request
    filestr = image_file.read()
    # Convert the file string to a NumPy array
    npimg = np.fromstring(filestr, np.uint8)
    
    # Decode the image using OpenCV
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # --- Critical Preprocessing Steps (must match your training script) ---
    # 1. Convert color from BGR (OpenCV default) to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 2. Resize to the target size
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    # 3. Normalize pixel values to be between 0 and 1
    img_normalized = img_resized / 255.0
    # 4. Expand dimensions to create a batch of 1 for the model
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

# Flask RESTful API Routes
@app.route('/', methods=['GET'])
def index():
    # Renders the main upload page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handles the image upload and returns the prediction as a JSON
    if model is None:
        return jsonify({'error': 'Model is not loaded properly'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    if file:
        try:
            # Preprocess the uploaded image
            processed_image = preprocess_image(file)
            
            # Make a prediction using the loaded model
            prediction_probs = model.predict(processed_image)
            
            # Get the index of the highest probability
            predicted_class_index = np.argmax(prediction_probs)
            # Get the confidence score
            confidence = np.max(prediction_probs) * 100
            
            # Map the index to the actual class name
            predicted_class_name = CATEGORIES[predicted_class_index]
            
            # Return the result
            return jsonify({
                'prediction': predicted_class_name,
                'confidence': f'{confidence:.2f}%'
            })
        except Exception as e:
            return jsonify({'error': f'An error occurred during prediction: {e}'}), 500


# Run the app
if __name__ == '__main__':
    # Use port 8080 to avoid conflicts with other services
    app.run(host='0.0.0.0', port=8080, debug=True)