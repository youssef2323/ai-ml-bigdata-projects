# German Traffic Sign Recognition Project
This project aims to recognize German traffic signs using Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN).

![Screenshot (289)](./Screenshots/screenshot(1).png)<br>
![Screenshot (289)](./Screenshots/screenshot(2).png)<br>
![Screenshot (289)](./Screenshots/screenshot(3).png)<br>



### Description
The project involves the following key steps and components:

### Data Preprocessing<br>
The traffic sign images are preprocessed to ensure they are suitable for training the models. This involves resizing the images, normalizing pixel values, and encoding labels.<br>

### Models Used<br>
Artificial Neural Network (ANN): Achieved an accuracy of 95%.<br>
Convolutional Neural Network (CNN): Achieved an accuracy of 99%.<br>
### Training and Evaluation<br>
Cross-Validation: Used to ensure the model's robustness and to prevent overfitting.<br>
Confusion Matrix: Generated to evaluate the performance of the models and to visualize the classification results.<br>
Loss and Accuracy Plots: Created to monitor the training process and to compare the performance of ANN and CNN.<br>
### Web Application<br>
A Flask server is used to serve the predictive model as a web application. This allows users to upload traffic sign images and get the predicted classification in real-time.<br>

### Libraries and Tools Used<br>
TensorFlow: For building and training the neural network models.<br>
Keras: A high-level API for TensorFlow to simplify model creation.<br>
NumPy: For numerical operations and handling arrays.<br>
Matplotlib: For plotting graphs and visualizing data.<br>
OS: For handling file operations.<br>
CV2: For image processing tasks.<br>
PIL: For additional image processing.<br>
Scikit-learn: For preprocessing, model evaluation, and cross-validation.<br>
Warnings: To suppress unnecessary warnings during execution.<br>
Flask: For creating the web application.<br>
By following these steps, the project demonstrates an effective approach to recognizing traffic signs using deep learning techniques. The comparison between ANN and CNN highlights the superior performance of CNN in image classification tasks.<br>

This project showcases the application of machine learning and image processing techniques to solve a real-world problem, achieving high accuracy and providing valuable insights into the performance of different neural network architectures.<br>
