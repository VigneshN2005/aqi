from flask import Flask, render_template, request
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

model = None
scaler = None


# 🔥 Safe + flexible model loading
def load_resources():
    global model, scaler

    if model is None:
        print("Loading model...")

        model = keras.models.load_model(
            "air_quality_model.h5",
            compile=False,
            safe_mode=False  # 🔥 bypass strict checks
        )

    if scaler is None:
        print("Loading scaler...")
        scaler = joblib.load("scaler.pkl")


def categorize(val):
    if val < 1:
        return "Good"
    elif val < 3:
        return "Moderate"
    else:
        return "Poor"


def advice(cat):
    if cat == "Good":
        return "Air quality is safe for all activities."
    elif cat == "Moderate":
        return "Sensitive groups should take precautions."
    else:
        return "Avoid outdoor exposure. Health risk detected."


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        load_resources()

        values = []
        for key in request.form:
            val = request.form.get(key)

            if val is None or val.strip() == "":
                return "Error: All input fields required"

            values.append(float(val))

        if len(values) != 12:
            return f"Error: Expected 12 inputs, got {len(values)}"

        data = np.array(values).reshape(1, -1)

        scaled = scaler.transform(data)
        seq = np.repeat(scaled, 10, axis=0).reshape(1, 10, -1)

        pred = model.predict(seq)[0][0]
        cat = categorize(pred)

        return render_template(
            'index.html',
            prediction=round(float(pred), 2),
            category=cat,
            advice=advice(cat),
            raw=list(data[0]),
            norm=list(scaled[0])
        )

    except Exception as e:
        print("ERROR:", str(e))
        return f"Internal Server Error: {str(e)}"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
