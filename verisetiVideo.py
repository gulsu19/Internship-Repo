import cv2
import os

def extract_frames(video_path, output_path, frame_interval=6):
    cap = cv2.VideoCapture(video_path)
    
    frame_number = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_number % frame_interval == 0:
            frame_filename = os.path.join(output_path, f"frame_{frame_number:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Frame {frame_number} saved as {frame_filename}")
        
        frame_number += 1
    
    cap.release()

# Video dosyası ve çıktı klasörü
video_path = "C:/Users/Gulsu/Desktop/Security Cam Shows Moment Earthquake Hits Taiwan.mp4"
output_path = "C:/Users/Gulsu/Desktop/frame3"
frame_interval = 12

if not os.path.exists(output_path):
    os.makedirs(output_path)

extract_frames(video_path, output_path, frame_interval)
