import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and details
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    return blurred_image

def detect_anomalies(image1, image2, threshold=30):
    # Resize both images to a common size
    common_size = (min(image1.shape[1], image2.shape[1]), min(image1.shape[0], image2.shape[0]))
    resized_image1 = cv2.resize(image1, common_size)
    resized_image2 = cv2.resize(image2, common_size)

    # Preprocess resized images to reduce lighting variations
    preprocessed_image1 = preprocess_image(resized_image1)
    preprocessed_image2 = preprocess_image(resized_image2)

    # Calculate absolute difference between preprocessed images
    diff = cv2.absdiff(preprocessed_image1, preprocessed_image2)

    # Apply thresholding to highlight anomalies
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    return thresholded


# Load images from camera (replace these paths with your actual image paths)
image1 = cv2.imread("C:\\Users\\Gulsu\\Desktop\\veri seti\\before38.png")
image2 = cv2.imread("C:\\Users\\Gulsu\\Desktop\\veri seti\\after38.png")

# Perform anomaly detection
anomaly_map = detect_anomalies(image1, image2, threshold=30)

# Plot the results
plt.figure(figsize=(10, 5))

plt.subplot(131)
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.title("Image 1")
plt.axis("off")

plt.subplot(132)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.title("Image 2")
plt.axis("off")

plt.subplot(133)
plt.imshow(anomaly_map, cmap="gray")
plt.title("Anomaly Map")
plt.axis("off")

plt.tight_layout()
plt.show()
