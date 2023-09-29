import cv2 as cv
import numpy as np

# Load YOLOv3 model
net = cv.dnn.readNet('C:\\Users\\Gulsu\\Downloads\\yolov3.cfg','C:\\Users\\Gulsu\\Downloads\\yolov3.weights')

# Load COCO classes
classes = []
with open('C:\\Users\\Gulsu\\Downloads\\coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]


outs = net.getUnconnectedOutLayers()
outs = [outs[i[0] - 1] for i in net.getUnconnectedOutLayers()]
output_layers = [layer_names[i[0] - 1] for i in outs]


# Load video
cap = cv.VideoCapture('C:\\Users\\Gulsu\\Desktop\\Earthquake Classroom Video.mp4')

while True:
    # Read frame from video
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to (416, 416)
    blob = cv.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

    # Pass frame through network
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process each output layer
    boxes = []
    confidences = []
    class_ids = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression
    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw boxes around detected humans
    for i in indices:
        i = i[0]
        x, y, w, h = boxes[i]
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display output
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
