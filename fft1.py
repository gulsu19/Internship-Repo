import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('C:\\Users\\Gulsu\\Desktop\\veri seti\\before37.png', cv2.IMREAD_GRAYSCALE)

# Resize the image dimensions to be powers of 2 for FFT processing
rows, cols = image.shape
padded_rows = cv2.getOptimalDFTSize(rows)
padded_cols = cv2.getOptimalDFTSize(cols)
padded_image = cv2.copyMakeBorder(image, 0, padded_rows - rows, 0, padded_cols - cols, cv2.BORDER_CONSTANT, value=0)

# Apply FFT for the image
fft_result = cv2.dft(np.float32(padded_image), flags=cv2.DFT_COMPLEX_OUTPUT)
fft_shifted = np.fft.fftshift(fft_result)
magnitude_spectrum = cv2.magnitude(fft_shifted[:, :, 0], fft_shifted[:, :, 1])

# Visualize the FFT result
plt.subplot(121), plt.imshow(padded_image, cmap='gray')
plt.title("Grayscale Image"), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(np.log(magnitude_spectrum), cmap='gray')
plt.title("FFT Result"), plt.xticks([]), plt.yticks([])

plt.show()
