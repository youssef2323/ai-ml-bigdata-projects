from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

model = load_model("model/my_model.h5")

scaler = joblib.load("model/scaler.pkl")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["T01"]),
            float(request.form["H01"]),
            float(request.form["T02"]),
            float(request.form["H02"]),
            float(request.form["T11"]),
            float(request.form["H11"]),
            float(request.form["T12"]),
            float(request.form["H12"]),
            float(request.form["T21"]),
            float(request.form["H21"]),
            float(request.form["T22"]),
            float(request.form["H22"]),
        ]
        print(f"Form data received: {features}")
        if len(features) != 12:
            raise ValueError(f"Expected 12 features, but got {len(features)}")

        input_data = np.array(features).reshape(1, 3, 4)
        input_data_scaled = scaler.transform(input_data.reshape(1, -1))

        input_data_scaled = input_data_scaled.reshape(1, 3, 4)
        predicted_output = model.predict(input_data_scaled)
        predicted_class = (predicted_output > 0.5).astype(int)
        prediction_label = "Normal" if predicted_class == 0 else "Faulty"

        return render_template("result.html", prediction=prediction_label)

    except Exception as e:
        return render_template("error.html", error_message=str(e))


if __name__ == "__main__":
    app.run(debug=True)
