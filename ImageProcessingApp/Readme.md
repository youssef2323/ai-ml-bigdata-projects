# Flask Image Processing Application

This project is a Flask web application for processing and analyzing images using various algorithms. The app allows users to upload an image, process it through different functionalities, and view/download the results.

---

## Examples of Results

### Original Image

![App Interface](./Screenshots/Screenshot_(467).png)

![Upload Image](./Screenshots/Screenshot_(469).png)


![Example 1](./Screenshots/Screenshot_(468).png)
![Example 2](./Screenshots/Screenshot_(471).png)
![Example 3](./Screenshots/Screenshot_(472).png)
![Example 4](./Screenshots/Screenshot_(473).png)
![Example 5](./Screenshots/Screenshot_(474).png)
![Example 6](./Screenshots/Screenshot_(475).png)
![Example 7](./Screenshots/Screenshot_(476).png)
![Example 8](./Screenshots/Screenshot_(477).png)



---

## Functionalities
- **Grayscale Conversion**: Converts the uploaded image to grayscale.
- **Thresholding**: Includes simple and advanced halftoning techniques.
- **Histogram Operations**:
  - Histogram visualization
  - Histogram equalization for contrast enhancement
- **Edge Detection**:
  - Sobel operator
  - Prewitt operator
  - Kirsch Compass Masks
  - Advanced edge detection using homogeneity, variance, and range operators
- **Filters**:
  - High-pass filter
  - Low-pass filter
  - Median filter
- **Image Operations**:
  - Addition of images
  - Subtraction of images
  - Image inversion
- **Segmentation**:
  - Manual histogram segmentation
  - Valley-based segmentation
  - Adaptive histogram segmentation
- **Difference of Gaussians (DoG)**: Edge enhancement using Gaussian filters.

---

## Technologies Used

| Technology    | Description                                        |
|---------------|----------------------------------------------------|
| **Flask**     | For building the web application.                 |
| **Pillow**    | Image handling and manipulation.                  |
| **OpenCV**    | Advanced image processing algorithms.             |
| **NumPy**     | Efficient numerical computations for processing.  |
| **Matplotlib**| Visualization and plotting histograms.            |
