from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

app = Flask(__name__)

# Lazy loading
model = None
scaler = None

def load_resources():
    global model, scaler
    if model is None:
        print("Loading model...")
        model = load_model("air_quality_model.keras", compile=False)
    if scaler is None:
        print("Loading scaler...")
        scaler = joblib.load("scaler.pkl")


# AQI Category
def categorize(val):
    if val < 1:
        return "Good"
    elif val < 3:
        return "Moderate"
    else:
        return "Poor"


# Health Advice
def advice(cat):
    if cat == "Good":
        return "Air quality is safe for all activities."
    elif cat == "Moderate":
        return "Sensitive groups should take precautions."
    else:
        return "Avoid outdoor exposure. Health risk detected."


# Home
@app.route('/')
def home():
    return render_template('index.html')


# Predict
@app.route('/predict', methods=['POST'])
def predict():
    try:
        load_resources()

        # 🔴 safer input handling
        values = []
        for key in request.form:
            val = request.form.get(key)

            if val is None or val.strip() == "":
                return "Error: All input fields must be filled"

            values.append(float(val))

        # 🔴 enforce correct feature size
        if len(values) != 12:
            return f"Error: Expected 12 inputs, got {len(values)}"

        data = np.array(values).reshape(1, -1)

        # 🔴 debug logs (very useful)
        print("Input:", data)

        scaled = scaler.transform(data)
        seq = np.repeat(scaled, 10, axis=0).reshape(1, 10, -1)

        pred = model.predict(seq)[0][0]
        cat = categorize(pred)

        return render_template(
            'index.html',
            prediction=round(pred, 2),
            category=cat,
            advice=advice(cat),
            raw=list(data[0]),
            norm=list(scaled[0])
        )

    except Exception as e:
        print("ERROR:", str(e))
        return f"Internal Server Error: {str(e)}"


# Entry
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
