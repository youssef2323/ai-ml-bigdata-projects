# Network Fault Detection App

A Flask-based web application designed to predict network faults using a state-of-the-art LSTM model. This application takes time-series data from sensors as input and predicts whether the network is "Normal" or "Faulty." The LSTM model achieved an impressive **99.9% accuracy** during evaluation.

---

---

## Screenshots

### Input Form
The main page allows users to input sensor readings.
![Input Form Screenshot](./Screenshots/Screenshot_(489).png)
![Input Form Screenshot](./Screenshots/Screenshot_(487).png)
![Input Form Screenshot](./Screenshots/Screenshot_(491).png)

### Prediction Result
Displays whether the network is "Normal" or "Faulty" based on the input.
![Prediction Result Screenshot](./Screenshots/Screenshot_(488).png)
![Prediction Result Screenshot](./Screenshots/Screenshot_(490).png)

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


