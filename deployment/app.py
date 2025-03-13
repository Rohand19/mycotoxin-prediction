from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import joblib

app = Flask(__name__)
model = load_model('don_prediction_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']  # Expecting a list of 448 reflectance values
    data = np.array(data).reshape(1, -1)
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return jsonify({'don_concentration_ppb': float(prediction[0][0])})

if __name__ == '__main__':
    app.run(debug=True)