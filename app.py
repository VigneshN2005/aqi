from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
 
app = Flask(__name__)
model = load_model("air_quality_model.keras")
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
 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
