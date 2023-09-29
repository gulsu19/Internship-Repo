import cv2
import numpy as np

# Path to the YOLO model's configuration file and trained weights
yolo_config = "C:\\Users\\Gulsu\\Downloads\\yolov3.cfg"
yolo_weights = "C:\\Users\\Gulsu\\Downloads\\yolov3.weights"

# Path to the class labels file
classes_file = "C:\\Users\\Gulsu\\Downloads\\coco.names"

# Index of the "human" class
human_class_index = 0  # In the COCO dataset, the index of the "human" class is 0

# Load class labels
with open(classes_file, 'r') as f:
    classes = f.read().strip().split('\n')

# Load the YOLO model
net = cv2.dnn.readNet(yolo_weights, yolo_config)

# Use GPU if available (uncomment these lines if GPU is supported)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Video capture
video_path = "C:\\Users\\Gulsu\\Desktop\\Earthquake Classroom Video.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width = frame.shape[:2]
    
    # Set the input image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Perform YOLO detection
    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)
    
    # Process detection results
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5 and class_id == 0:  # Detect only humans
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Draw a rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow("Video", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
