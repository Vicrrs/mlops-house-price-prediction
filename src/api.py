# src/api.py
from flask import Flask, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)
model = load("models/model.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([list(data.values())])
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
