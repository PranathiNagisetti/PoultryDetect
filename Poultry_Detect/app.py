import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from keras.preprocessing.image import load_img, img_to_array
import gdown
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model path
MODEL_PATH = "healthy_vs_rotten.h5"

# Download model from Google Drive if not present
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1orvFuJXAk-Bc-4pn3tdnv--eVz--F5IK"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (same order as training)
labels = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

# Function to preprocess and predict
def get_model_prediction(image_path):
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img) / 255.0   # Normalize
    x = np.expand_dims(x, axis=0)   # Add batch dimension

    predictions = model.predict(x, verbose=0)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    return labels[predicted_class], float(confidence)

# Home Page
@app.route('/')
def index():
    return render_template("index.html")

# About Page
@app.route('/about')
def about():
    return render_template("blog-single.html")

# Contact Page
@app.route('/contact')
def contact():
    return render_template("blog.html")

# Upload Page
@app.route('/predict-page')
def predict_page():
    return render_template("portfolio-details.html")

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['pc_image']

        if file.filename == '':
            return "No file uploaded"

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        prediction, confidence = get_model_prediction(filepath)

        return render_template(
            "contact.html",
            predict=prediction,
            confidence=round(confidence * 100, 2),
            image_name=filename
        )

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)