from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from ml.classifier import PlastiTraceClassifier

app = Flask(__name__)
CORS(app)  # Allow frontend to call API

# Load model
classifier = PlastiTraceClassifier("models/plastitrace.pth")

@app.route('/api/classify', methods=['POST'])
def classify():
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        # Read image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Classify
        result = classifier.predict_from_bgr(img)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)  # Ganti dari 5000 ke 5001