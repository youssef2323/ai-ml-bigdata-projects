from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


app = Flask(__name__)

# Setup folder for storing uploaded and processed images
UPLOAD_FOLDER = "./static/uploads"
PROCESSED_FOLDER = "./static/processed_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


# Helper functions for image processing
def convert_to_grayscale(image):
    image_array = np.array(image)
    if len(image_array.shape) == 3:
        grayscale = (
            0.2989 * image_array[:, :, 0]
            + 0.5870 * image_array[:, :, 1]
            + 0.1140 * image_array[:, :, 2]
        )
        return grayscale.astype(np.uint8)
    else:
        return image_array


def apply_threshold(gray_scale_image):
    threshold = np.mean(gray_scale_image)
    binary_image = np.where(gray_scale_image >= threshold, 255, 0)
    return binary_image.astype(np.uint8)


def simple_halftoning(grayscale_image, threshold=128):
    binary_image = np.where(grayscale_image >= threshold, 255, 0)
    return binary_image.astype(np.uint8)


def Advanced_halftoning(grayscale_image):
    rows, cols = grayscale_image.shape
    output_image = grayscale_image.copy()

    for i in range(rows):
        for j in range(cols):
            old_pixel = output_image[i, j]
            new_pixel = 255 if old_pixel > 128 else 0
            output_image[i, j] = new_pixel

            error = old_pixel - new_pixel

            if j + 1 < cols:
                output_image[i, j + 1] += (7 / 16) * error
            if i + 1 < rows and j - 1 >= 0:
                output_image[i + 1, j - 1] += (3 / 16) * error
            if i + 1 < rows:
                output_image[i + 1, j] += (5 / 16) * error
            if i + 1 < rows and j + 1 < cols:
                output_image[i + 1, j + 1] += (1 / 16) * error

    return np.clip(output_image, 0, 255).astype(np.uint8)


def calculate_histogram(grayscale_image):
    histogram = np.zeros(256, dtype=int)
    for value in grayscale_image.flatten():
        histogram[value] += 1
    return histogram


def histogram_equalization(grayscale_image):

    histogram = calculate_histogram(grayscale_image)

    cumulative_histogram = [0] * len(histogram)
    cumulative_histogram[0] = histogram[0]
    for i in range(1, len(histogram)):
        cumulative_histogram[i] = cumulative_histogram[i - 1] + histogram[i]

    dm = 256
    area = grayscale_image.shape[0] * grayscale_image.shape[1]

    normalized_cumulative_histogram = []
    for value in cumulative_histogram:
        normalized_value = round((dm - 1) * (value / area))
        normalized_cumulative_histogram.append(normalized_value)

    equalized_image = np.zeros_like(grayscale_image)
    rows, cols = grayscale_image.shape
    for i in range(rows):
        for j in range(cols):
            equalized_image[i, j] = int(
                normalized_cumulative_histogram[grayscale_image[i, j]]
            )

    return (
        equalized_image.astype(np.uint8),
        histogram,
        cumulative_histogram,
        normalized_cumulative_histogram,
    )


def sobel_operator(grayscale_image):
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    rows, cols = grayscale_image.shape
    edge_image = np.zeros_like(grayscale_image)
    magnitudes = []

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = grayscale_image[i - 1 : i + 2, j - 1 : j + 2]

            gx = np.sum(Gx * region)
            gy = np.sum(Gy * region)

            magnitude = np.sqrt(gx**2 + gy**2)
            magnitudes.append(magnitude)

    threshold = np.mean(magnitudes)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = grayscale_image[i - 1 : i + 2, j - 1 : j + 2]

            gx = np.sum(Gx * region)
            gy = np.sum(Gy * region)

            magnitude = np.sqrt(gx**2 + gy**2)
            edge_image[i, j] = 255 if magnitude > threshold else 0

    return edge_image.astype(np.uint8)


def prewitt_operator(grayscale_image):
    Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    Gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    rows, cols = grayscale_image.shape
    edge_image = np.zeros_like(grayscale_image)
    magnitudes = []

    # Compute gradient magnitudes
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = grayscale_image[i - 1 : i + 2, j - 1 : j + 2]

            gx = np.sum(Gx * region)
            gy = np.sum(Gy * region)

            magnitude = np.sqrt(gx**2 + gy**2)
            magnitudes.append(magnitude)

    threshold = np.mean(magnitudes)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = grayscale_image[i - 1 : i + 2, j - 1 : j + 2]

            gx = np.sum(Gx * region)
            gy = np.sum(Gy * region)

            magnitude = np.sqrt(gx**2 + gy**2)
            edge_image[i, j] = 255 if magnitude > threshold else 0

    return edge_image.astype(np.uint8)


def kirsch_compass_masks(grayscale_image, threshold=128):
    kirsch_masks = [
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  # North
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),  # North-East
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),  # East
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),  # South-East
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),  # South
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),  # South-West
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),  # West
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),  # North-West
    ]

    rows, cols = grayscale_image.shape
    edge_image = np.zeros_like(grayscale_image)
    direction_image = np.zeros_like(grayscale_image, dtype=int)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = grayscale_image[i - 1 : i + 2, j - 1 : j + 2]

            responses = [np.sum(mask * region) for mask in kirsch_masks]

            max_response = max(responses)
            max_direction = responses.index(max_response)

            edge_image[i, j] = 255 if max_response > threshold else 0
            direction_image[i, j] = max_direction

    return edge_image.astype(np.uint8), direction_image


def homogeneity_operator(grayscale_image, threshold=128):

    rows, cols = grayscale_image.shape
    edge_image = np.zeros_like(grayscale_image)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighborhood = grayscale_image[i - 1 : i + 2, j - 1 : j + 2]
            center_pixel = grayscale_image[i, j]

            differences = np.abs(neighborhood - center_pixel)

            max_difference = np.max(differences)

            edge_image[i, j] = 255 if max_difference > threshold else 0

    return edge_image.astype(np.uint8)


def difference_operator(grayscale_image, threshold=128):

    rows, cols = grayscale_image.shape
    edge_image = np.zeros_like(grayscale_image)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = grayscale_image[i - 1 : i + 2, j - 1 : j + 2]

            differences = [
                abs(region[0, 0] - region[2, 2]),
                abs(region[2, 0] - region[0, 2]),
                abs(region[1, 0] - region[1, 2]),
                abs(region[0, 1] - region[2, 1]),
            ]

            max_difference = max(differences)

            edge_image[i, j] = 255 if max_difference > threshold else 0

    return edge_image.astype(np.uint8)


def difference_of_guassians(image, kernel1, kernel2):

    smoothed1 = cv2.filter2D(image, -1, kernel1)
    smoothed2 = cv2.filter2D(image, -1, kernel2)

    dog_result = smoothed1 - smoothed2

    return np.clip(dog_result, 0, 255).astype(np.uint8), smoothed1, smoothed2


kernel7 = np.array(
    [
        [0, 0, -1, -1, -1, 0, 0],
        [0, -2, -3, -3, -3, -2, 0],
        [-1, -3, 5, 5, 5, -3, -1],
        [-1, -3, 5, 16, 5, -3, -1],
        [-1, -3, 5, 5, 5, -3, -1],
        [0, -2, -3, -3, -3, -2, 0],
        [0, 0, -1, -1, -1, 0, 0],
    ]
)

kernel9 = np.array(
    [
        [0, 0, 0, -1, -1, -1, 0, 0, 0],
        [0, -2, -3, -3, -3, -3, -3, -2, 0],
        [0, -3, -2, -1, -1, -1, -2, -3, 0],
        [-1, -3, -1, 9, 9, 9, -1, -3, -1],
        [-1, -3, -1, 9, 19, 9, -1, -3, -1],
        [-1, -3, -1, 9, 9, 9, -1, -3, -1],
        [0, -3, -2, -1, -1, -1, -2, -3, 0],
        [0, -2, -3, -3, -3, -3, -3, -2, 0],
        [0, 0, 0, -1, -1, -1, 0, 0, 0],
    ]
)


def contrast_based_edge_detection_with_edge_detector(image, threshold=15):

    smoothing_mask = (1 / 9) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    smoothed_image = cv2.filter2D(image, -1, smoothing_mask)

    edge_detector_mask = np.array([[-1, 0, -1], [0, 4, 0], [-1, 0, -1]])

    edge_response = cv2.filter2D(smoothed_image, -1, edge_detector_mask)

    edges = np.where(edge_response > threshold, 255, 0)

    return edges.astype(np.uint8)


def variance_edge_detector(image, region_size=3, threshold=50):
    pad = region_size // 2

    padded_image = np.pad(image, pad, mode="constant", constant_values=0)

    rows, cols = image.shape
    edge_image = np.zeros_like(image)

    for i in range(pad, rows + pad):
        for j in range(pad, cols + pad):

            region = padded_image[i - pad : i + pad + 1, j - pad : j + pad + 1]

            mean_intensity = np.mean(region)

            variance = np.mean((region - mean_intensity) ** 2)

            edge_image[i - pad, j - pad] = 255 if variance > threshold else 0

    return edge_image.astype(np.uint8)


def range_edge_detector(image, region_size=3, threshold=50):
    pad = region_size // 2

    padded_image = np.pad(image, pad, mode="constant", constant_values=0)

    rows, cols = image.shape
    edge_image = np.zeros_like(image)

    for i in range(pad, rows + pad):
        for j in range(pad, cols + pad):

            region = padded_image[i - pad : i + pad + 1, j - pad : j + pad + 1]

            intensity_range = np.max(region) - np.min(region)

            edge_image[i - pad, j - pad] = 255 if intensity_range > threshold else 0

    return edge_image.astype(np.uint8)


def high_pass_filter(image):

    high_pass_mask = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    filtered_image = cv2.filter2D(image, -1, high_pass_mask)

    filtered_image = np.clip(filtered_image, 0, 255)

    return filtered_image.astype(np.uint8)


def low_pass_filter(image):

    low_pass_mask = (1 / 6) * np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]])

    #     filtered_image = manual_convolution(image, low_pass_mask)
    filtered_image = cv2.filter2D(image, -1, low_pass_mask)

    filtered_image = np.clip(filtered_image, 0, 255)

    return filtered_image.astype(np.uint8)


def median_filter(image, region_size=3):

    pad = region_size // 2

    padded_image = np.pad(image, pad, mode="constant", constant_values=0)

    rows, cols = image.shape
    filtered_image = np.zeros_like(image)

    for i in range(pad, rows + pad):
        for j in range(pad, cols + pad):

            region = padded_image[i - pad : i + pad + 1, j - pad : j + pad + 1]

            median_value = np.median(region)

            filtered_image[i - pad, j - pad] = median_value

    return filtered_image.astype(np.uint8)


def add_imagess(image):
    image_copy = image.copy()
    added_image = image + image_copy
    return np.clip(added_image, 0, 255).astype(np.uint8)


def subtract_imagess(image):
    image_copy = image.copy()
    subtracted_image = image_copy - image

    return np.clip(subtracted_image, 0, 255).astype(np.uint8)


def invert_image(image):
    invert_image = 255 - image
    return invert_image.astype(np.uint8)


def manual_histogram_segmentation(image, thresholds):

    thresholds = sorted(thresholds)
    segmented_image = np.zeros_like(image)

    for i, threshold in enumerate(thresholds):
        segmented_image[image <= threshold] = (i + 1) * 50
        image = np.where(image > threshold, image, 0)

        segmented_image[image > 0] = (len(thresholds) + 1) * 50

        return segmented_image


def histogram_peak_thresholding(image):

    histogram = np.zeros(256, dtype=int)
    for pixel in image.ravel():  # Flatten the image into a 1D array
        histogram[pixel] += 1

    peaks = []
    for i in range(1, len(histogram) - 1):
        if histogram[i] > histogram[i - 1] and histogram[i] > histogram[i + 1]:
            peaks.append(i)

    peaks = sorted(peaks, key=lambda x: histogram[x], reverse=True)

    if len(peaks) < 2:
        raise ValueError("Not enough peaks found in histogram!")
    background_peak = peaks[0]
    object_peak = peaks[1]

    threshold = (background_peak + object_peak) // 2

    binary_image = ((image > threshold) * 255).astype(np.uint8)

    return binary_image, threshold, histogram, background_peak, object_peak


def manual_peak_detection(histogram):
    peaks = []

    for i in range(1, len(histogram) - 1):
        if histogram[i] > histogram[i - 1] and histogram[i] > histogram[i + 1]:
            peaks.append(i)

    return peaks


def valley_based_segmentation(image):

    histogram = calculate_histogram(image)

    all_peaks = manual_peak_detection(histogram)

    prominent_peaks = sorted(all_peaks, key=lambda x: histogram[x], reverse=True)

    if len(prominent_peaks) < 2:
        raise ValueError("Not enough peaks detected to calculate a valley.")
    peak1, peak2 = prominent_peaks[:2]

    start, end = min(peak1, peak2), max(peak1, peak2)
    print(
        f"Selected Peaks -> Peak 1: {peak1}, Peak 2: {peak2}, Range: {start} to {end}"
    )  # Debugging Output

    valley_range = histogram[start : end + 1]
    if len(valley_range) == 0:
        raise ValueError("Invalid valley range. Check peaks and histogram.")
    valley = np.argmin(valley_range) + start
    print(f"Valley Detected: {valley}")

    segmented_image = np.zeros_like(image)
    segmented_image[image <= valley] = 50
    segmented_image[image > valley] = 255

    return segmented_image, histogram, (peak1, peak2), valley


def adaptive_histogram_segmentation(image):

    histogram = calculate_histogram(image)

    all_peaks = manual_peak_detection(histogram)

    prominent_peaks = sorted(all_peaks, key=lambda x: histogram[x], reverse=True)

    if len(prominent_peaks) < 2:
        raise ValueError("Not enough peaks detected for adaptive segmentation.")

    peak1, peak2 = prominent_peaks[:2]

    start, end = min(peak1, peak2), max(peak1, peak2)
    valley = np.argmin(histogram[start : end + 1]) + start

    segmented_image = np.zeros_like(image)
    segmented_image[image <= valley] = 50  # Background
    segmented_image[image > valley] = 255  # Object

    background_mean = np.mean(image[segmented_image == 50])
    object_mean = np.mean(image[segmented_image == 255])

    new_threshold = (background_mean + object_mean) // 2

    final_segmented_image = np.zeros_like(image)
    final_segmented_image[image <= new_threshold] = 50  # Background
    final_segmented_image[image > new_threshold] = 255  # Object

    return {
        "final_segmented_image": final_segmented_image,
        "histogram": histogram,
        "all_peaks": all_peaks,
        "prominent_peaks": (peak1, peak2),
        "valley": valley,
        "background_mean": background_mean,
        "object_mean": object_mean,
        "new_threshold": new_threshold,
    }


@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return "No image part", 400

    file = request.files["image"]
    if file.filename == "":
        return "No selected file", 400

    if file:
        filename = file.filename
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)

        # Show the uploaded image
        return render_template(
            "index.html", original_image=filename, show_uploaded=True
        )


@app.route("/process/<function_name>/<filename>", methods=["GET"])
def process_image(function_name, filename):
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image = Image.open(image_path)
    processed_filename = ""
    processed_image = None
    first_pass_filename = None  # Default to None to avoid UnboundLocalError
    final_segmented_filename = None  # Default to None

    if function_name == "grayscale":
        processed_image = convert_to_grayscale(image)
        processed_filename = f"grayscale_{filename}"
    elif function_name == "threshold":
        grayscale_image = convert_to_grayscale(image)
        processed_image = apply_threshold(grayscale_image)
        processed_filename = f"binary_{filename}"
    elif function_name == "simple_halftoning":
        grayscale_image = convert_to_grayscale(image)
        processed_image = simple_halftoning(grayscale_image)
        processed_filename = f"simple_halftoned_{filename}"
    elif function_name == "advanced_halftoning":
        grayscale_image = convert_to_grayscale(image)
        processed_image = Advanced_halftoning(grayscale_image)
        processed_filename = f"advanced_halftoned_{filename}"
    elif function_name == "histogram":
        grayscale_image = convert_to_grayscale(image)
        histogram = calculate_histogram(grayscale_image)
        # Save the histogram plot
        plt.figure(figsize=(10, 5))
        plt.bar(range(256), histogram, color="gray")
        plt.title("Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        processed_filename = f"histogram_{filename}.png"
        histogram_path = os.path.join(PROCESSED_FOLDER, processed_filename)
        plt.savefig(histogram_path)
        plt.close()

        return render_template(
            "index.html",
            original_image=filename,
            processed_image=processed_filename,
            show_uploaded=True,
        )

    elif function_name == "histogram_equalization":
        grayscale_image = convert_to_grayscale(image)
        (
            equalized_image,
            histogram,
            cumulative_histogram,
            normalized_cumulative_histogram,
        ) = histogram_equalization(grayscale_image)

        # Save the equalized image
        processed_image = equalized_image
        processed_filename = f"equalized_{filename}"
        Image.fromarray(equalized_image).save(
            os.path.join(PROCESSED_FOLDER, processed_filename)
        )

    elif function_name == "sobel_operator":
        grayscale_image = convert_to_grayscale(image)
        processed_image = sobel_operator(grayscale_image)  # Sobel operator processing
        if processed_image is None:  # Check if Sobel operator fails to process
            print(f"Sobel operator failed for {filename}")
            processed_image = np.zeros_like(grayscale_image)  # Fallback to black image
        processed_filename = f"sobel_{filename}"

    elif function_name == "prewitt_operator":
        grayscale_image = convert_to_grayscale(image)
        processed_image = prewitt_operator(grayscale_image)
        processed_filename = f"perwitt_{filename}"

    elif function_name == "kirsch_compass_masks":
        grayscale_image = convert_to_grayscale(image)
        processed_image, direction_image = kirsch_compass_masks(grayscale_image)
        processed_filename = f"kirsch_{filename}"

    elif function_name == "homogeneity_operator":
        grayscale_image = convert_to_grayscale(image)
        processed_image = homogeneity_operator(grayscale_image)
        processed_filename = f"homogenity_{filename}"

    elif function_name == "difference_operator":
        grayscale_image = convert_to_grayscale(image)
        processed_image = difference_operator(grayscale_image)
        processed_filename = f"difference_{filename}"

    elif function_name == "difference_of_guassians":
        grayscale_image = convert_to_grayscale(image)
        processed_image, smoothed1, smoothed2 = difference_of_guassians(
            grayscale_image, kernel7, kernel9
        )
        processed_filename = f"difference_of_guassians_{filename}"

    elif function_name == "contrast_based_edge_detection_with_edge_detector":
        grayscale_image = convert_to_grayscale(image)
        processed_image = contrast_based_edge_detection_with_edge_detector(
            grayscale_image, threshold=15
        )
        processed_filename = f"contrast_based_{filename}"

    elif function_name == "variance_edge_detector":
        grayscale_image = convert_to_grayscale(image)
        processed_image = variance_edge_detector(grayscale_image)
        processed_filename = f"variance_{filename}"

    elif function_name == "range_edge_detector":
        grayscale_image = convert_to_grayscale(image)
        processed_image = range_edge_detector(grayscale_image)
        processed_filename = f"range_{filename}"

    elif function_name == "high_pass_filter":
        grayscale_image = convert_to_grayscale(image)
        processed_image = high_pass_filter(grayscale_image)
        processed_filename = f"rangehigh_pass_filter_{filename}"

    elif function_name == "low_pass_filter":
        grayscale_image = convert_to_grayscale(image)
        processed_image = low_pass_filter(grayscale_image)
        processed_filename = f"low_pass_filter_{filename}"

    elif function_name == "median_filter":
        grayscale_image = convert_to_grayscale(image)
        processed_image = median_filter(grayscale_image)
        processed_filename = f"median_filter_{filename}"

    elif function_name == "add_imagess":
        grayscale_image = convert_to_grayscale(image)
        processed_image = add_imagess(grayscale_image)
        processed_filename = f"add_imagess_{filename}"

    elif function_name == "subtract_imagess":
        grayscale_image = convert_to_grayscale(image)
        processed_image = subtract_imagess(grayscale_image)
        processed_filename = f"subtract_imagess_{filename}"

    elif function_name == "invert_image":
        grayscale_image = convert_to_grayscale(image)
        processed_image = invert_image(grayscale_image)
        processed_filename = f"invert_image_{filename}"

    elif function_name == "manual_histogram_segmentation":
        grayscale_image = convert_to_grayscale(image)
        processed_image = manual_histogram_segmentation(
            grayscale_image, thresholds=[125, 255]
        )
        processed_filename = f"manual_histogram_segmentation_{filename}"

    elif function_name == "histogram_peak_thresholding":
        grayscale_image = convert_to_grayscale(image)
        processed_image, threshold, histogram, bg_peak, obj_peak = (
            histogram_peak_thresholding(grayscale_image)
        )
        processed_filename = f"histogram_peak_thresholding{filename}"

    elif function_name == "valley_based_segmentation":
        grayscale_image = convert_to_grayscale(image)
        processed_image, histogram, peaks, valley = valley_based_segmentation(
            grayscale_image
        )
        processed_filename = f"valley_based_segmentation_{filename}"

    elif function_name == "adaptive_histogram_segmentation":
        grayscale_image = convert_to_grayscale(image)

        # Perform adaptive histogram segmentation
        results = adaptive_histogram_segmentation(grayscale_image)

        # Extract values from the results
        first_pass_segmentation = (grayscale_image <= results["valley"]).astype(
            np.uint8
        ) * 255
        final_segmented_image = results["final_segmented_image"]

        # Generate filenames for both images
        first_pass_filename = f"first_pass_segmentation_{filename}"
        final_segmented_filename = f"final_segmented_image_{filename}"

        # Save the processed images
        first_pass_path = os.path.join(PROCESSED_FOLDER, first_pass_filename)
        final_segmented_path = os.path.join(PROCESSED_FOLDER, final_segmented_filename)

        Image.fromarray(first_pass_segmentation).save(first_pass_path)
        Image.fromarray(final_segmented_image).save(final_segmented_path)

        # Return the results as filenames
        processed_filename = (
            first_pass_filename  # Optionally choose which image to show first
        )

    # Save the processed image (for all cases except histogram)
    if function_name != "histogram":
        if processed_image is not None:
            processed_image_path = os.path.join(PROCESSED_FOLDER, processed_filename)
            Image.fromarray(processed_image).save(processed_image_path)
        else:
            print(f"Error: processed_image is None for {filename}")

    return render_template(
        "index.html",
        original_image=filename,
        processed_image=processed_filename,
        first_pass_segmentation=first_pass_filename,  # Add first pass
        final_segmented_image=final_segmented_filename,  # Add final pass
        show_uploaded=True,
    )


@app.route("/static/processed_images/<filename>")
def send_processed_image(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/process/halftoning/<filename>", methods=["GET"])
def halftoning_image(filename):
    # Load the image
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image = Image.open(image_path)
    grayscale_image = convert_to_grayscale(image)

    # Apply both halftoning functions
    simple_binary = simple_halftoning(grayscale_image)
    advanced_binary = Advanced_halftoning(grayscale_image)

    # Save processed images
    simple_filename = f"simple_halftoned_{filename}"
    advanced_filename = f"advanced_halftoned_{filename}"

    simple_path = os.path.join(PROCESSED_FOLDER, simple_filename)
    advanced_path = os.path.join(PROCESSED_FOLDER, advanced_filename)

    Image.fromarray(simple_binary).save(simple_path)
    Image.fromarray(advanced_binary).save(advanced_path)

    # Render both processed images in the template
    return render_template(
        "index.html",
        original_image=filename,
        simple_halftoned_image=simple_filename,
        advanced_halftoned_image=advanced_filename,
        show_uploaded=True,
    )


@app.route("/process/Edge_Detection/<filename>", methods=["GET"])
def Edge_Detection(filename):
    # Load the image
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image = Image.open(image_path)
    grayscale_image = convert_to_grayscale(image)

    # Apply both halftoning functions
    sobel = sobel_operator(grayscale_image)
    perwit = prewitt_operator(grayscale_image)
    kirsch, direction_image = kirsch_compass_masks(grayscale_image)

    # Save processed images
    sobel_filename = f"sobel_{filename}"
    perwit_filename = f"perwit_{filename}"
    kirsch_filename = f"kirsch_{filename}"

    sobel_path = os.path.join(PROCESSED_FOLDER, sobel_filename)
    perwit_path = os.path.join(PROCESSED_FOLDER, perwit_filename)
    kirsch_path = os.path.join(PROCESSED_FOLDER, kirsch_filename)

    Image.fromarray(sobel).save(sobel_path)
    Image.fromarray(perwit).save(perwit_path)
    Image.fromarray(kirsch).save(kirsch_path)

    # Render both processed images in the template
    return render_template(
        "index.html",
        original_image=filename,
        sobel_image=sobel_filename,
        perwitt_image=perwit_filename,
        kirsch_image=kirsch_filename,
        show_uploaded=True,
    )


@app.route("/process/Advanced_Edge_Detection/<filename>", methods=["GET"])
def Advanced_Edge_Detection(filename):
    # Load the image
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image = Image.open(image_path)
    grayscale_image = convert_to_grayscale(image)

    # Apply both halftoning functions
    homog = homogeneity_operator(grayscale_image)
    diff = difference_operator(grayscale_image)
    dog, smoothed1, smoothed2 = difference_of_guassians(
        grayscale_image, kernel7, kernel9
    )
    contrast = contrast_based_edge_detection_with_edge_detector(grayscale_image)
    var = variance_edge_detector(grayscale_image)
    rang = range_edge_detector(grayscale_image)

    # Save processed images
    homog_filename = f"homog_{filename}"
    diff_filename = f"diff_{filename}"
    dog_filename = f"dog_{filename}"
    contrast_filename = f"contrast_{filename}"
    var_filename = f"var_{filename}"
    rang_filename = f"rang_{filename}"

    homog_path = os.path.join(PROCESSED_FOLDER, homog_filename)
    diff_path = os.path.join(PROCESSED_FOLDER, diff_filename)
    dog_path = os.path.join(PROCESSED_FOLDER, dog_filename)
    contrast_path = os.path.join(PROCESSED_FOLDER, contrast_filename)
    var_path = os.path.join(PROCESSED_FOLDER, var_filename)
    rang_path = os.path.join(PROCESSED_FOLDER, rang_filename)

    Image.fromarray(homog).save(homog_path)
    Image.fromarray(diff).save(diff_path)
    Image.fromarray(dog).save(dog_path)
    Image.fromarray(contrast).save(contrast_path)
    Image.fromarray(var).save(var_path)
    Image.fromarray(rang).save(rang_path)

    # Render both processed images in the template
    return render_template(
        "index.html",
        original_image=filename,
        homog_image=homog_filename,
        diff_image=diff_filename,
        dog_image=dog_filename,
        contrast_image=contrast_filename,
        var_image=var_filename,
        rang_image=rang_filename,
        show_uploaded=True,
    )


@app.route("/process/Filtering/<filename>", methods=["GET"])
def Filtering(filename):
    # Load the image
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image = Image.open(image_path)
    grayscale_image = convert_to_grayscale(image)

    # Apply both halftoning functions
    high_mask = high_pass_filter(grayscale_image)
    low_mask = low_pass_filter(grayscale_image)
    median_mask = median_filter(grayscale_image)

    # Save processed images
    high_filename = f"high_{filename}"
    low_filename = f"low_{filename}"
    median_filename = f"median_{filename}"

    high_path = os.path.join(PROCESSED_FOLDER, high_filename)
    low_path = os.path.join(PROCESSED_FOLDER, low_filename)
    median_path = os.path.join(PROCESSED_FOLDER, median_filename)

    Image.fromarray(high_mask).save(high_path)
    Image.fromarray(low_mask).save(low_path)
    Image.fromarray(median_mask).save(median_path)

    # Render both processed images in the template
    return render_template(
        "index.html",
        original_image=filename,
        high_image=high_filename,
        low_image=low_filename,
        median_image=median_filename,
        show_uploaded=True,
    )


@app.route("/process/Operations/<filename>", methods=["GET"])
def Operations(filename):
    # Load the image
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image = Image.open(image_path)
    grayscale_image = convert_to_grayscale(image)

    # Apply both halftoning functions
    add_op = add_imagess(grayscale_image)
    sub_op = subtract_imagess(grayscale_image)
    inv_op = invert_image(grayscale_image)

    # Save processed images
    add_filename = f"add_{filename}"
    sub_filename = f"sub_{filename}"
    inv_filename = f"inv_{filename}"

    add_path = os.path.join(PROCESSED_FOLDER, add_filename)
    sub_path = os.path.join(PROCESSED_FOLDER, sub_filename)
    inv_path = os.path.join(PROCESSED_FOLDER, inv_filename)

    Image.fromarray(add_op).save(add_path)
    Image.fromarray(sub_op).save(sub_path)
    Image.fromarray(inv_op).save(inv_path)

    # Render both processed images in the template
    return render_template(
        "index.html",
        original_image=filename,
        add_image=add_filename,
        sub_image=sub_filename,
        inv_image=inv_filename,
        show_uploaded=True,
    )


@app.route("/process/Segmenetation/<filename>", methods=["GET"])
def Segmentations(filename):
    # Load the image
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image = Image.open(image_path)
    grayscale_image = convert_to_grayscale(image)

    # Apply both halftoning functions
    manual = manual_histogram_segmentation(grayscale_image, thresholds=[125, 255])
    peak, threshold, histogram, bg_peak, obj_peak = histogram_peak_thresholding(
        grayscale_image
    )

    valleybased, histogram, peaks, valley = valley_based_segmentation(grayscale_image)

    # Perform adaptive histogram segmentation
    results = adaptive_histogram_segmentation(grayscale_image)

    # Extract values from the results
    first_pass_segmentation = (grayscale_image <= results["valley"]).astype(
        np.uint8
    ) * 255
    final_segmented_image = results["final_segmented_image"]

    add_seg = results["final_segmented_image"]

    # Save processed images
    manual_filename = f"manual_{filename}"
    peak_filename = f"peak_{filename}"
    valley_filename = f"valley_{filename}"
    add_seg_filename = f"add_seg_{filename}"
    first_pass_filename = f"first_pass_segmentation_{filename}"
    final_segmented_filename = f"final_segmented_image_{filename}"

    manual_path = os.path.join(PROCESSED_FOLDER, manual_filename)
    peak_path = os.path.join(PROCESSED_FOLDER, peak_filename)
    valley_path = os.path.join(PROCESSED_FOLDER, valley_filename)
    add_seg_path = os.path.join(PROCESSED_FOLDER, add_seg_filename)
    first_pass_path = os.path.join(PROCESSED_FOLDER, first_pass_filename)
    final_segmented_path = os.path.join(PROCESSED_FOLDER, final_segmented_filename)

    Image.fromarray(manual).save(manual_path)
    Image.fromarray(peak).save(peak_path)
    Image.fromarray(valleybased).save(valley_path)
    Image.fromarray(add_seg).save(add_seg_path)
    Image.fromarray(first_pass_segmentation).save(first_pass_path)
    Image.fromarray(final_segmented_image).save(final_segmented_path)

    # Render both processed images in the template
    return render_template(
        "index.html",
        original_image=filename,
        manual_image=manual_filename,
        peak_image=peak_filename,
        valley_image=valley_filename,
        first_pass_segmentation=first_pass_filename,  # Add first pass
        final_segmented_image=final_segmented_filename,
        add_seg_image=add_seg_filename,
        show_uploaded=True,
    )


if __name__ == "__main__":
    app.run(debug=True)
