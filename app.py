import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from scipy.stats import kurtosis, skew
import onnxruntime as ort

app = Flask(__name__)

# Load the ONNX model
MODEL_PATH = 'mo_model.onnx'  # Update with your ONNX model path
session = ort.InferenceSession(MODEL_PATH)

# Sensor labels
sensor_labels = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'rot_x', 'rot_y', 'rot_z', 'heart']

# Function to calculate only the specified features
def calculate_features(axis_data, axis_name):
    features = {
        f"{axis_name}_mean": np.mean(axis_data),
        f"{axis_name}_max": np.max(axis_data),
        f"{axis_name}_min": np.min(axis_data),
        f"{axis_name}_std": np.std(axis_data),
        f"{axis_name}_energy": np.sum(np.square(axis_data)),
        f"{axis_name}_kurtosis": kurtosis(axis_data, fisher=True),
        f"{axis_name}_skew": skew(axis_data),
        f"{axis_name}_rms": np.sqrt(np.mean(np.square(axis_data))),
        f"{axis_name}_rss": np.sqrt(np.sum(np.square(axis_data))),
        f"{axis_name}_area": np.sum(axis_data),
        f"{axis_name}_abs_area": np.sum(np.abs(axis_data)),
        f"{axis_name}_abs_mean": np.mean(np.abs(axis_data)),
        f"{axis_name}_range": np.ptp(axis_data),
        f"{axis_name}_lower_q": np.percentile(axis_data, 25),
        f"{axis_name}_median": np.median(axis_data),
        f"{axis_name}_upper_q": np.percentile(axis_data, 75),
        f"{axis_name}_mad": np.mean(np.abs(axis_data - np.mean(axis_data))),
    }
    return features

# Function to process time-series data
def process_time_series(time_series_data, window_size=3):
    rows = []
    num_samples = len(next(iter(time_series_data.values())))  # Get the number of samples

    for start in range(0, num_samples - window_size + 1, window_size):
        window_features = {}
        for sensor, values in time_series_data.items():
            if sensor in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:  # Process only specified sensors
                window = values[start:start + window_size]
                sensor_features = calculate_features(window, sensor)
                window_features.update(sensor_features)

        # Add derived features
        acc_x = np.mean(time_series_data['acc_x'][start:start + window_size])
        acc_y = np.mean(time_series_data['acc_y'][start:start + window_size])
        acc_z = np.mean(time_series_data['acc_z'][start:start + window_size])

        window_features["angle_x"] = np.arctan2(acc_y, acc_x)
        window_features["angle_y"] = np.arctan2(acc_z, acc_x)
        window_features["angle_z"] = np.arctan2(acc_z, acc_y)
        window_features["mag"] = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        window_features["heart"] = np.mean(time_series_data['heart'][start:start + window_size])

        rows.append(window_features)

    return pd.DataFrame(rows)

# Endpoint to handle live predictions with direct data input
@app.route("/live_predict", methods=["POST"])
def live_predict():
    try:
        # Parse the incoming data from the request
        input_data = request.json

        # Ensure the required fields are present in the request
        required_fields = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'rot_x', 'rot_y', 'rot_z', 'heart']
        for field in required_fields:
            if field not in input_data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Prepare the time-series data directly from the input
        time_series_data = {
            "acc_x": input_data["acc_x"],
            "acc_y": input_data["acc_y"],
            "acc_z": input_data["acc_z"],
            "gyro_x": input_data["gyro_x"],
            "gyro_y": input_data["gyro_y"],
            "gyro_z": input_data["gyro_z"],
            "rot_x": input_data["rot_x"],
            "rot_y": input_data["rot_y"],
            "rot_z": input_data["rot_z"],
            "heart": input_data["heart"]
        }

        # Process the time-series data (window_size is fixed to 3)
        features_df = process_time_series(time_series_data, window_size=3)

        # Prepare features for prediction
        feature_vector = features_df.values.astype(np.float32)

        # Validate model input dimensions
        input_name = session.get_inputs()[0].name
        model_input_shape = session.get_inputs()[0].shape
        if feature_vector.shape[1] != model_input_shape[1]:
            raise ValueError(f"Feature dimension mismatch: Features have {feature_vector.shape[1]} columns, "
                             f"but the model expects {model_input_shape[1]} columns.")

        # Make predictions for each time window
        predictions = [float(session.run(None, {input_name: row.reshape(1, -1)})[0][0]) for row in feature_vector]

        # Map predictions to emotions
        emotion_map = {-1: "sad", 0: "neutral", 1: "happy"}
        emotions = [emotion_map.get(pred, "unknown") for pred in predictions]

        return jsonify({"predictions": predictions, "emotions": emotions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route("/", methods=["GET"])
def home():
    return "ONNX Model API for Live Data is running with window_size=3!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
