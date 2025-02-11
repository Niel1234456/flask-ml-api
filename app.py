from flask import Flask, request, jsonify
import tensorflow as tf 
from tensorflow import keras 
import numpy as np
from PIL import Image # type: ignore
import io
import gdown  # For downloading from Google Drive
import os

app = Flask(__name__)

# Google Drive File ID
FILE_ID = "1Qu4j-09HJoRJX3Ldbxk6Un6L2YqOrl1G"  # Replace with your actual File ID
MODEL_PATH = "best_model.keras"

# Function to download the model if it doesn't exist
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
    else:
        print("Model already exists, skipping download.")

# Download the model before loading
download_model()

# Load your trained model
model = keras.models.load_model('C:/Users/chris/Desktop/PROJECT/best_model.keras')

def preprocess_image(image):
    image = image.resize((150, 150))  # Adjust based on your model's input size
    image = np.array(image) / 255.0
    if image.ndim == 3:  # Ensure that the image has 3 channels
        image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    image = Image.open(io.BytesIO(file.read()))
    processed_image = preprocess_image(image)

    try:
        predictions = model.predict(processed_image)
        prediction_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100
    except Exception as e:
        return jsonify({'error': str(e)})

    return jsonify({
        'prediction': int(prediction_class),
        'confidence': float(confidence)
    })

if __name__ == '__main__':
    app.run(debug=True)
