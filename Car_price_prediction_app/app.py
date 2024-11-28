from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the models and scaler
model_linear_svr = joblib.load("linear_svr_model.pkl")
model_DT = joblib.load("DT_model.pkl")
model_GB = joblib.load("GB.pkl")  # Gradient Boosting model
scaler = joblib.load("scaler.pkl")
x_columns = joblib.load("x_columns.pkl")  # Load the feature names


def predict_car_price(mark, model, fuel, year, mileage, vol_engine, x_columns, scaler):
    try:
        brand_index = np.where(x_columns == mark)[0][0]
        model_index = np.where(x_columns == model)[0][0]
        fuel_index = np.where(x_columns == fuel)[0][0]
    except IndexError:
        print("One or more categorical inputs do not match the feature names.")
        return "Invalid input."

    features = np.zeros(len(x_columns))
    features[0] = year
    features[1] = mileage
    features[2] = vol_engine

    if brand_index >= 0:
        features[brand_index] = 1
    if model_index >= 0:
        features[model_index] = 1
    if fuel_index >= 0:
        features[fuel_index] = 1

    features_scaled = scaler.transform([features])

    # Getting predictions from all models
    prediction_svr = model_linear_svr.predict(features_scaled)[0]
    prediction_dt = model_DT.predict(features_scaled)[0]
    prediction_gb = model_GB.predict(features_scaled)[0]

    return {
        "LinearSVR": f"{prediction_svr:.2f}",
        "DecisionTree": f"{prediction_dt:.2f}",
        "GradientBoosting": f"{prediction_gb:.2f}",
    }


@app.route("/", methods=["GET", "POST"])
def index():
    predicted_prices = {}
    if request.method == "POST":
        mark = request.form["mark"]
        model = request.form["model"]
        fuel = request.form["fuel"]
        year = int(request.form["year"])
        mileage = int(request.form["mileage"])
        vol_engine = int(request.form["vol_engine"])

        predicted_prices = predict_car_price(
            mark, model, fuel, year, mileage, vol_engine, x_columns, scaler
        )

    return render_template("index.html", predicted_prices=predicted_prices)


if __name__ == "__main__":
    app.run(debug=True)
