import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import cv2
import base64
import time

app = Flask(__name__)

# Load model
MODEL_PATH = 'models/Baseline.keras'
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print(f"✅ Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print(f"⚠️ Model file not found at {MODEL_PATH}")

load_model()

# EMNIST Balanced: 47 classes (digits + letters merged case)
LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e',
    'f', 'g', 'h', 'n', 'q', 'r', 't'
]

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
        image_data = data['image']

        # Decode base64 image
        encoded_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # Resize to 28x28
        img_resized = cv2.resize(img, (28, 28))
        
        # Invert if background is white
        if np.mean(img_resized) > 127: 
            img_resized = 255 - img_resized

        # Save debug
        os.makedirs('request', exist_ok=True)
        timestamp = int(time.time() * 1000)
        cv2.imwrite(f'request/{timestamp}_processed.png', img_resized)

        # Normalize
        img_norm = img_resized.astype('float32') / 255.0
        img_input = img_norm.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(img_input, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        label = get_label(class_idx)

        # Top 5 predictions
        top5_idx = np.argsort(prediction[0])[-5:][::-1]
        top5 = [
            {
                'label': get_label(i),
                'prob': float(prediction[0][i])
            } for i in top5_idx
        ]

        return jsonify({
            'label': label,
            'confidence': confidence,
            'top5': top5
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)