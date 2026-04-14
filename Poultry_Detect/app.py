import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from keras.preprocessing.image import load_img, img_to_array
import gdown
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Model path
MODEL_PATH = "healthy_vs_rotten.h5"

# Download model from Google Drive if not present
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1orvFuJXAk-Bc-4pn3tdnv--eVz--F5IK"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
labels = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

# Function to preprocess and predict (FROM MEMORY)
from PIL import Image

def get_model_prediction(file):
    img = Image.open(file.stream).convert("RGB")
    img = img.resize((224, 224))

    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    predictions = model.predict(x, verbose=0)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    return labels[predicted_class], float(confidence)
# Routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("blog-single.html")

@app.route('/contact')
def contact():
    return render_template("blog.html")

@app.route('/predict-page')
def predict_page():
    return render_template("portfolio-details.html")

# Prediction Route (UPDATED)
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['pc_image']

    if file.filename == '':
        return "No file uploaded"

    prediction, confidence = get_model_prediction(file)

    return render_template(
        "contact.html",
        predict=prediction,
        confidence=round(confidence * 100, 2)
    )

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
