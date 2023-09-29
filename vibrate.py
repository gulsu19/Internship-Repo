import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("C:\\Users\\Gulsu\\Desktop\\Earthquake Classroom Video.mp4")
bgSub = cv2.createBackgroundSubtractorMOG2()

vibration_values = []  # List to store the vibration values over time

plt.ion()  # Turn on interactive mode for matplotlib

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)

    mask = bgSub.apply(frame, learningRate=50/100)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Calculate the percentage of pixels that are part of the foreground (vibrating)
    vibration_percentage = (np.sum(mask) / (mask.shape[0] * mask.shape[1] * 255)) * 100

    vibration_values.append(vibration_percentage)

    cv2.imshow("Video Feed", frame)
    cv2.imshow("Background Subtraction", mask)

    # Plot the vibration graph
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(range(len(vibration_values)), vibration_values, label='Vibration Percentage')
    plt.xlabel('Time (frames)')
    plt.ylabel('Vibration Percentage')
    plt.title('Vibration Over Time')
    plt.legend()
    plt.grid(True)

    plt.pause(0.01)  # Pause to update the graph

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' key to exit
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()  # Turn off interactive mode when done