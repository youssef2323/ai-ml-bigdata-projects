# Network Fault Detection App

A Flask-based web application designed to predict network faults using a state-of-the-art LSTM model. This application takes time-series data from sensors as input and predicts whether the network is "Normal" or "Faulty." The LSTM model achieved an impressive **99.9% accuracy** during evaluation.

---

## Features

- **Accurate Fault Detection**: Predicts network faults with a high degree of accuracy.
- **User-Friendly Interface**: Simple and intuitive web interface for inputting sensor data.
- **Real-Time Predictions**: Immediate feedback on network status.
- **Error Handling**: Displays clear error messages for invalid inputs.
- **Scalable Backend**: Built using Flask and TensorFlow for seamless integration.

---

## How It Works

1. **Input Data**: Enter 12 sensor readings (temperature and humidity for different time steps) in the input form.
2. **Prediction**: The LSTM model processes the data and predicts whether the network is "Normal" or "Faulty."
3. **Results**: The prediction result is displayed instantly on the result page.

---

## Screenshots

### Input Form
The main page allows users to input sensor readings.
![Input Form Screenshot](screenshots/input_form.png)

### Prediction Result
Displays whether the network is "Normal" or "Faulty" based on the input.
![Prediction Result Screenshot](screenshots/prediction_result.png)

### Error Handling
Provides clear feedback for invalid inputs or prediction errors.
![Error Page Screenshot](screenshots/error_page.png)

---

## Technologies Used

- **Backend**:
  - Flask
  - TensorFlow/Keras
  - Scikit-learn
- **Frontend**:
  - HTML, CSS, JavaScript
- **Data Analysis**:
  - NumPy
  - Pandas
- **Visualization**:
  - Matplotlib 


