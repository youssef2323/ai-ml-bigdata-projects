from flask import Flask, request, render_template, send_from_directory
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("./model.h5")
categories = [str(i) for i in range(43)]
TEST_FOLDER = "./Test"


def process_image(image_path):
    img = Image.open(image_path)
    img = img.resize((30, 30))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.utils.normalize(img_array, axis=1)
    return img_array


def predict_image_category(image_path):
    img_array = process_image(image_path)
    predictions = model.predict(img_array)
    predicted_class_index = tf.argmax(predictions[0]).numpy()
    predicted_class_name = categories[predicted_class_index]
    return predicted_class_name


@app.route("/test/<filename>")
def test(filename):
    return send_from_directory(TEST_FOLDER, filename)


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            filename = (
                image_file.filename
            )  # You might want to ensure the filename is secure
            image_location = os.path.join(TEST_FOLDER, filename)
            image_file.save(image_location)
            predicted_class = predict_image_category(image_location)
            return render_template(
                "result.html", prediction=predicted_class, image_loc=filename
            )
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
