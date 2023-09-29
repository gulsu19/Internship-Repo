import cv2
import numpy as np

# Giriş ve çıkış bölgelerinin koordinatları (x1, y1, x2, y2)
enter_line = [(50, 0), (50, 720)]
exit_line = [(600, 0), (600, 720)]

# Giriş ve çıkış sayaçları
enter_count = 0
exit_count = 0

# Video dosyasının adı
video_path = 'C:\\Users\\Gulsu\\Desktop\\Earthquake Classroom Video.mp4'
cap = cv2.VideoCapture(video_path)

# OpenCV CascadeClassifier kullanarak yüz tespiti
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Yüz tespiti
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # İnsanları tespit etme
    for (x, y, w, h) in faces:
        center_x = x + w // 2
        center_y = y + h // 2
        
        if enter_line[0][0] < center_x < enter_line[1][0] and enter_line[0][1] < center_y < enter_line[1][1]:
            enter_count += 1
        elif exit_line[0][0] < center_x < exit_line[1][0] and exit_line[0][1] < center_y < exit_line[1][1]:
            exit_count += 1
    
    # Giriş ve çıkış sayılarını frame üzerine yazdırma
    cv2.putText(frame, f'Enter: {enter_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Exit: {exit_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Görüntüyü kaydetme
    cv2.imwrite('output_frame.jpg', frame)
    
    # 'q' tuşuna basıldığında döngüyü sonlandırma
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırakma
cap.release()
cv2.destroyAllWindows()
