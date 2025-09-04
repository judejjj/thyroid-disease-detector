import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request
import os

# -------------------------------
# Load Phase 1 model
# -------------------------------
MODEL_PATH = "thyroid_model_clinical.pkl"
DATASET_PATH = "thyroid_clinical_balanced.csv"

with open(MODEL_PATH, 'rb') as f:
    components = pickle.load(f)

ensemble = components['ensemble']
scaler = components['scaler']
numeric_features = components['numeric_features']
class_names = components['class_names']

core_features = [
    'age', 'sex', 'on thyroxine', 'query on thyroxine', 
    'on antithyroid medication', 'sick', 'pregnant', 
    'thyroid surgery', 'I131 treatment', 'query hypothyroid',
    'query hyperthyroid', 'lithium', 'goitre', 'tumor', 
    'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI'
]

# -------------------------------
# Load dataset stats for intelligent filling
# -------------------------------
df_stats = pd.read_csv(DATASET_PATH)
numeric_means = df_stats[numeric_features].mean()
categorical_modes = df_stats[core_features].mode().iloc[0]

# -------------------------------
# Flask setup
# -------------------------------
app = Flask(__name__)

def preprocess_patient_data(patient_data):
    """Ensure all core features are present, fill intelligently"""
    processed = {}
    for feat in core_features:
        if feat in patient_data:
            processed[feat] = patient_data[feat]
        else:
            # Fill numeric with mean, categorical/binary with mode
            if feat in numeric_features:
                processed[feat] = numeric_means[feat]
            else:
                processed[feat] = categorical_modes[feat]
    return pd.DataFrame([processed])

def predict_patient(df):
    """Make prediction using ensemble model"""
    df[numeric_features] = scaler.transform(df[numeric_features])
    pred = ensemble.predict(df)[0]
    prob = ensemble.predict_proba(df)[0]
    return {
        'predicted_class': int(pred),
        'condition': class_names[pred],
        'confidence': float(max(prob)),
        'all_probabilities': {class_names[i]: float(p) for i, p in enumerate(prob)}
    }

@app.route("/", methods=['GET', 'POST'])
def index():
    prediction_result = None
    patient_input = {}
    
    if request.method == 'POST':
        # Handle CSV upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.csv'):
                input_df = pd.read_csv(file)
                # Take the first row as patient
                patient_input = input_df.iloc[0].to_dict()
            # TODO: PDF parsing can be added here
            
        # Handle manual form input
        for feat in core_features:
            value = request.form.get(feat)
            if value is not None and value != "":
                try:
                    patient_input[feat] = float(value)
                except:
                    patient_input[feat] = value
        
        # Preprocess & predict
        processed_df = preprocess_patient_data(patient_input)
        prediction_result = predict_patient(processed_df)
    
    return render_template("index.html",
                           features=core_features,
                           patient_input=patient_input,
                           prediction=prediction_result)

if __name__ == "__main__":
    app.run(debug=True)
