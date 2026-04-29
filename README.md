# Air Quality Prediction App 🌍

A simple web application that predicts air quality levels using Machine Learning. This app takes air quality sensor readings and tells you if the air is **Good**, **Moderate**, or **Poor** for outdoor activities.

---

## 📋 Table of Contents
1. [How It Works](#how-it-works)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Running the App](#running-the-app)
5. [How to Use](#how-to-use)
6. [Understanding the Code](#understanding-the-code)

---

## 🎯 How It Works

### Simple Explanation:
Think of this app like a **weather prediction system for air quality**:

1. **You enter air quality measurements** (like PM2.5, NO2, CO levels)
2. **The app processes your data** using a trained AI model
3. **The model predicts if air is safe** for outdoor activities
4. **You get advice** on whether it's safe to go outside

### The 3 Steps:

**Step 1: Data Normalization** 📊
- Raw sensor readings vary widely (0-500 scale)
- We convert them to a standard scale (0-1) using a "scaler"
- This helps the AI model understand the data better

**Step 2: AI Prediction** 🧠
- A trained neural network (deep learning model) analyzes the normalized data
- It was trained on thousands of real air quality samples
- It outputs a prediction score

**Step 3: Categorization & Advice** 💬
- Score < 1 → **Good** - Safe for all activities
- Score 1-3 → **Moderate** - Sensitive groups should be careful
- Score ≥ 3 → **Poor** - Avoid going outside

---

## 📁 Project Structure

```
air_quality_app/
│
├── app.py                          # Main Flask web server (handles everything)
├── air_quality_model.keras         # Pre-trained AI model file
├── scaler.pkl                      # Data normalizer file
│
├── templates/
│   └── index.html                  # Web page (what you see in browser)
│
├── static/
│   └── style.css                   # Styling for the web page
│
├── requirements.txt                # List of all dependencies
└── README.md                       # This file!
```

### File Explanations:

| File | Purpose |
|------|---------|
| `app.py` | The brain - handles all logic, predictions, and web requests |
| `air_quality_model.keras` | Saved AI model - makes predictions based on patterns it learned |
| `scaler.pkl` | Data converter - standardizes raw sensor readings |
| `index.html` | Frontend - what users see and interact with |
| `style.css` | Styling - makes the website look nice |

---

## 🚀 Installation

### Prerequisites:
- Python 3.10 or higher
- pip (Python package manager)

### Step-by-Step:

1. **Clone or download this project**
   ```
   git clone https://github.com/VigneshN2005/aqi.git
   cd aqi
   ```

2. **Install all dependencies**
   ```
   pip install -r requirements.txt
   ```

   This installs:
   - `flask` - Web server framework
   - `numpy` - Math and data processing
   - `tensorflow` - AI/Machine Learning
   - `joblib` - Loading saved models
   - `scikit-learn` - Data preprocessing

---

## ▶️ Running the App

### Start the server:
```
python app.py
```

### You'll see:
```
* Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

### Open in browser:
- Go to: **http://127.0.0.1:5000**
- You'll see the input form

---

## 💻 How to Use the App

1. **Enter Air Quality Values**
   - Enter various sensor readings (the exact values depend on your data)
   - Example: PM2.5: 35, NO2: 45, CO: 0.8, etc.

2. **Click "Predict"**
   - The app sends your data to the server
   - AI model processes it

3. **See Results**
   - Prediction score (e.g., 1.45)
   - Category (Good/Moderate/Poor)
   - Health advice specific to the air quality level

---

## 🔍 Understanding the Code

### How `app.py` Works (Line by Line):

```python
from flask import Flask, render_template, request
```
- Import Flask to create a web server
- Import tools to handle web pages and form data

```python
from tensorflow.keras.models import load_model
```
- Load our AI model file

```python
import joblib
```
- Load the data scaler (normalizer)

```python
app = Flask(__name__)
model = load_model("air_quality_model.keras")
scaler = joblib.load("scaler.pkl")
```
- Start web server
- Load the pre-trained AI model
- Load the data scaler

```python
def categorize(val):
    if val < 1:
        return "Good"
    elif val < 3:
        return "Moderate"
    else:
        return "Poor"
```
- Convert prediction score to human-readable category
- This is just basic if-else logic

```python
def advice(cat):
    if cat == "Good":
        return "Air quality is safe for all activities."
    ...
```
- Give health recommendations based on category

```python
@app.route('/')
def home():
    return render_template('index.html')
```
- When user visits the website, show them the form page

```python
@app.route('/predict', methods=['POST'])
def predict():
    values = [float(x) for x in request.form.values()]  # Get user input
    data = np.array(values).reshape(1, -1)              # Format for model
    scaled = scaler.transform(data)                     # Normalize the data
    seq = np.repeat(scaled, 10, axis=0).reshape(1, 10, -1)  # Reshape for model
    pred = model.predict(seq)[0][0]                     # Get prediction
    cat = categorize(pred)                              # Categorize result
    return render_template('index.html', ...)           # Send results back
```
- This is the main prediction function
- Takes user input → normalizes it → runs AI model → returns result

---

## 🤖 How the AI Model Works (Simple Explanation)

The AI model is like a **sophisticated pattern recognizer**:

1. **Training Phase (already done)**
   - Model saw 1000s of real air quality readings
   - Learned patterns: "When these sensors read like this, air is usually good/bad"

2. **Prediction Phase (what we do)**
   - New sensor reading comes in
   - Model says: "Based on patterns I learned, this looks like: 1.5 score"
   - We tell user: "That's MODERATE air quality"

**Why do we reshape data?**
```python
seq = np.repeat(scaled, 10, axis=0).reshape(1, 10, -1)
```
- The model expects time-series data (like data over time)
- We repeat the single reading 10 times to simulate a time series
- This is how the model was trained to accept data

---

## 📊 Data Flow Diagram

```
User Input (HTML Form)
        ↓
    app.py /predict route
        ↓
Extract values from form
        ↓
Normalize with scaler.pkl
        ↓
Reshape data for model
        ↓
AI Model predicts (air_quality_model.keras)
        ↓
Categorize result (Good/Moderate/Poor)
        ↓
Generate health advice
        ↓
Display result on website
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'flask'` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: scaler.pkl not found` | Make sure all files are in same directory |
| Port 5000 already in use | Run `python app.py --port 5001` or kill other process |
| Model fails to load | Ensure `air_quality_model.keras` is not corrupted |

---

## 📚 Learning Resources

- **Flask Basics**: https://flask.palletsprojects.com/
- **Neural Networks**: https://www.youtube.com/watch?v=aircAruvnKk
- **Machine Learning**: https://scikit-learn.org/stable/

---

## 📝 License

This project is open source. Feel free to use and modify!

---

## 👤 Author

Created for learning and demonstration purposes.

---

## ❓ Questions?

If you're a beginner and confused about any part:
1. Read the **"Understanding the Code"** section
2. Check the **"How It Works"** section
3. Search for the specific term on Google or ChatGPT

**Remember**: Every expert programmer started as a beginner! 🚀
