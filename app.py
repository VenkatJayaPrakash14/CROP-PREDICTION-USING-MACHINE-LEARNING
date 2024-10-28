

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import json

# Initialize Flask app
from flask import Flask, render_template

app = Flask(__name__)






# Load the Random Forest model and feature columns
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('features_data.json', 'r') as json_file:
    features_data = json.load(json_file)
    feature_columns = features_data['columns']

# Load label encoder to map encoded labels back to crop names
with open('.ipynb_checkpoints\\label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form
    try:
        # Create an empty input data array with zeros
        input_data = pd.Series(np.zeros(len(feature_columns)), index=feature_columns)
        
        # Populate input data with form values
        input_data['pin_code'] = float(data.get('Pincode'))
        input_data['N'] = float(data.get('Nitrogen'))
        input_data['P'] = float(data.get('Phosporus'))
        input_data['K'] = float(data.get('Potassium'))
        input_data['temperature'] = float(data.get('Temperature'))
        input_data['humidity'] = float(data.get('Humidity'))
        input_data['ph'] = float(data.get('Ph'))
        input_data['rainfall'] = float(data.get('Rainfall'))

        # Predict probabilities for each crop
        probabilities = model.predict_proba([input_data])[0]
        crop_probabilities = {crop: prob for crop, prob in zip(label_encoder.classes_, probabilities)}
        
        # Sort crops by highest probability and apply a threshold (e.g., 15%)
        threshold = 0.15
        recommended_crops = [crop for crop, prob in sorted(crop_probabilities.items(), key=lambda x: x[1], reverse=True) if prob >= threshold]

        # Return the top recommended crops
        if recommended_crops:
            result = ", ".join(recommended_crops)
        else:
            result = "No suitable crops found based on the input parameters."

    except Exception as e:
        result = f"Error in prediction: {e}"

    # Render result in the template
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

import os
print("Current Working Directory:", os.getcwd())   
