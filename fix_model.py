from tensorflow import keras

# Load old model safely
model = keras.models.load_model("air_quality_model.keras", compile=False)

# Save in new compatible format
model.save("air_quality_model_fixed.keras")

print("Model successfully fixed and saved!")