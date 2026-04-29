from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

app = Flask(__name__)

# 🔹 Lazy loading variables
model = None
scaler = None

# 🔹 Load model & scaler only when needed
def load_resources():
    global model, scaler
    if model is None:
        print("Loading model...")
        model = load_model("air_quality_model.keras", compile=False)
    if scaler is None:
        print("Loading scaler...")
        scaler = joblib.load("scaler.pkl")


# 🔹 AQI Category
def categorize(val):
    if val < 1:
        return "Good"
    elif val < 3:
        return "Moderate"
    else:
        return "Poor"


# 🔹 Health Advice
def advice(cat):
    if cat == "Good":
        return "Air quality is safe for all activities."
    elif cat == "Moderate":
        return "Sensitive groups should take precautions."
    else:
        return "Avoid outdoor exposure. Health risk detected."


# 🔹 Home Route
@app.route('/')
def home():
    return render_template('index.html')


# 🔹 Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    load_resources()  # ✅ ensures model loads only when needed

    values = [float(x) for x in request.form.values()]
    data = np.array(values).reshape(1, -1)

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


# 🔹 Entry point (only for local run, not used by gunicorn but safe to keep)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
