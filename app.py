import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import cv2
import base64
import time

app = Flask(__name__)

# Load model
# Load model
MODEL_PATH = 'experiments/models/ResNet_Deep.h5'
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"✅ Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print(f"⚠️ Model file not found at {MODEL_PATH}")

load_model()

# Mapping for EMNIST Balanced (37 classes - No Digits)
# 0-25: Uppercase (A-Z)
# 26-36: Lowercase (a, b, d, e, f, g, h, n, q, r, t)
import string

# Define the label list exactly as valid for EMNIST Balanced
# Note: EMNIST balanced merges some upper/lower classes that look identical (c, i, j, k, l, m, o, p, s, u, v, w, x, y, z)
# The remaining 11 lowercase letters are:
LOWERCASE_SUBSET = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
LABELS = list(string.ascii_uppercase) + LOWERCASE_SUBSET

def get_label(index):
    if 0 <= index < len(LABELS):
        return LABELS[index]
    return "?"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.json
        image_data = data['image'] # base64 string
        rotate_angle = data.get('rotate', 0) # 0, 90, 180, 270
        is_mirror = data.get('mirror', False)

        # Decode base64
        # Format is "data:image/png;base64,....."
        encoded_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # Preprocess
        # Resize to 28x28
        img_resized = cv2.resize(img, (28, 28))
        
        # Check if background is white
        if np.mean(img_resized) > 127: 
            img_resized = 255 - img_resized

        # Apply transformations requested by UI
        if is_mirror:
            img_resized = cv2.flip(img_resized, 1) # Flip horizontal

        if rotate_angle == 90:
            img_resized = cv2.rotate(img_resized, cv2.ROTATE_90_CLOCKWISE)
        elif rotate_angle == 180:
            img_resized = cv2.rotate(img_resized, cv2.ROTATE_180)
        elif rotate_angle == 270:
            img_resized = cv2.rotate(img_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # DEBUG: Save request images
        os.makedirs('request', exist_ok=True)
        timestamp = int(time.time() * 1000)
        cv2.imwrite(f'request/{timestamp}_original.png', img)
        cv2.imwrite(f'request/{timestamp}_processed.png', img_resized)

        # Normalize to 0-1
        img_norm = img_resized.astype('float32') / 255.0
        
        # Reshape to (1, 28, 28, 1)
        img_input = img_norm.reshape(1, 28, 28, 1)

        # Debug: Encode processed image to show in UI
        # Scale back to 0-255 for display
        img_debug = (img_norm * 255).astype(np.uint8)
        _, buffer = cv2.imencode('.png', img_debug)
        debug_base64 = base64.b64encode(buffer).decode('utf-8')

        # Predict
        prediction = model.predict(img_input)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        label = get_label(class_idx)

        # Get all probabilities
        all_probs = {get_label(i): float(prediction[0][i]) for i in range(len(LABELS))}

        return jsonify({
            'label': label,
            'confidence': f"{confidence*100:.2f}%",
            'all_probs': all_probs,
            'debug_image': f"data:image/png;base64,{debug_base64}"
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
