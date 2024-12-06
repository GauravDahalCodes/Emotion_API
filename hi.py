from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/sensor_data", methods=["GET"])
def sensor_data():
    # Example time-series data
    data = {
        "acc_x": [-0.4, -0.3, 0.2, 0.1, -0.1, 0.05],
        "acc_y": [0.1, 0.2, -0.3, -0.4, 0.05, 0.1],
        "acc_z": [0.5, 0.4, -0.2, -0.1, 0.15, 0.2],
        "gyro_x": [-9.1, -8.2, 7.3, 6.4, -6.5, 7.0],
        "gyro_y": [14.3, -2.2, -15.2, -10.5, 5.0, -4.3],
        "gyro_z": [-20.3, -11.1, 7.2, 6.5, -6.0, 5.5],
        "rot_x": [2.5, 2.4, 1.2, 1.0, 1.1, 1.3],
        "rot_y": [1.8, 1.7, 1.6, 1.5, 1.4, 1.3],
        "rot_z": [2.0, 1.9, 1.5, 1.2, 1.1, 1.0],
        "heart": [90, 91, 89, 88, 87, 86]
    }
    return jsonify(data)

if __name__ == "__main__":
    app.run(port=4000, debug=True)
