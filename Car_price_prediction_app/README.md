# Car Price Prediction Project
This project aims to predict car prices using various regression models including Linear SVR, Decision Tree, and Gradient Boosting. The project also includes a Flask server to serve the predictive model as a web application.
![Screenshot (295)](./Screenshots/screenshot(1).png)<br>
![Screenshot (295)](./Screenshots/screenshot(2).png
)<br>


## Description
The project involves the following key steps and components:<br>

## Data Preprocessing
The car data is preprocessed to ensure it is suitable for training the models. This involves handling missing values, encoding categorical variables, and scaling numerical features.<br>

## Models Used
 Linear SVR: A Support Vector Regression model with a linear kernel.<br>
 Decision Tree Regressor: A decision tree-based model for regression tasks.<br>
 Gradient Boosting Regressor: An ensemble model that builds multiple decision trees and combines their predictions for better accuracy.<br>
### Training and Evaluation
Cross-Validation: Used to ensure the model's robustness and to prevent overfitting.<br>
Grid Search: Performed to find the optimal hyperparameters for the models.<br>
Evaluation Metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared (R²) score, and Mean Squared Logarithmic Error (MSLE) are used to evaluate model performance.<br>
PCA: Principal Component Analysis is used for dimensionality reduction and to visualize data in lower dimensions.<br>
### Web Application
A Flask server is used to serve the predictive model as a web application. This allows users to input car features and get the predicted price in real-time.<br>

## Libraries and Tools Used
Pandas: For data manipulation and analysis.<br>
NumPy: For numerical operations and handling arrays.<br>
Matplotlib: For plotting graphs and visualizing data.<br>
Seaborn: For statistical data visualization.<br>
Plotly: For interactive data visualization.<br>
Scikit-learn: For building and evaluating machine learning models.<br>
Scipy: For statistical analysis.<br>
Flask: For creating the web application.<br>
Warnings: To suppress unnecessary warnings during execution.<br>

By following these steps, the project demonstrates an effective approach to predicting car prices using machine learning techniques and serving the model via a web application. The comparison between Linear SVR, Decision Tree, and Gradient Boosting highlights the strengths and weaknesses of each model in regression tasks.<br>

This project showcases the application of machine learning, data preprocessing, and web development techniques to solve a real-world problem, achieving high accuracy and providing valuable insights into the performance of different regression models.<br>



