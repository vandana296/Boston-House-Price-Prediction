

import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Get absolute path of the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build full paths of the model and scaler files
model_path = os.path.join(BASE_DIR, "regmodel.pkl")
scaler_path = os.path.join(BASE_DIR, "scaling.pkl")

# Load The Model using absolute paths
regmodel = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)

    arr = np.array(list(data.values())).reshape(1, -1)
    new_data = scaler.transform(arr)
    output = regmodel.predict(new_data)

    print(output[0])
    return jsonify({"prediction": float(output[0])})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
    
@app.route('/predict', methods=['POST'])
def predict():
    values = [float(x) for x in request.form.values()]   # get all form inputs
    final = scaler.transform([values])                   # scale features
    pred = regmodel.predict(final)[0]                    # get prediction
    return render_template('home.html', prediction=round(pred, 2))
