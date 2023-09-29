import cv2
import numpy as np

# Video dosyasının adı
video_path = 'C:\\Users\\Gulsu\\Desktop\\Earthquake Classroom Video.mp4'

# Giriş ve çıkış bölgelerinin koordinatları (x1, y1, x2, y2)
enter_line = [(50, 0), (50, 720)]
exit_line = [(600, 0), (600, 720)]

# Giriş ve çıkış sayaçları
enter_count = 0
exit_count = 0

# Background subtraction modeli oluşturma
fgbg = cv2.createBackgroundSubtractorMOG2()

# Video dosyasını açma
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Frame üzerinde arka plan çıkarma işlemi
    fgmask = fgbg.apply(frame)
    
    # Giriş ve çıkış bölgelerini çizme
    cv2.line(frame, enter_line[0], enter_line[1], (0, 255, 0), 2)
    cv2.line(frame, exit_line[0], exit_line[1], (0, 0, 255), 2)
    
    # Giriş ve çıkış bölgelerini oluşturma
    enter_region = np.array(enter_line)
    exit_region = np.array(exit_line)
    
    # Giriş ve çıkış bölgeleri ile maske oluşturma
    enter_mask = cv2.polylines(np.zeros_like(fgmask), [enter_region], isClosed=False, color=255, thickness=2)
    exit_mask = cv2.polylines(np.zeros_like(fgmask), [exit_region], isClosed=False, color=255, thickness=2)
    
    # Giriş ve çıkış bölgeleri ile maskeyi and işlemi
    enter_masked = cv2.bitwise_and(fgmask, enter_mask)
    exit_masked = cv2.bitwise_and(fgmask, exit_mask)
    
    # Giriş ve çıkış bölgelerindeki beyaz pikselleri sayma
    enter_pixel_count = np.sum(enter_masked == 255)
    exit_pixel_count = np.sum(exit_masked == 255)
    
    # Giriş ve çıkış sayılarını güncelleme
    if enter_pixel_count > 50:  # Eşik değeri ayarlayarak optimize edebilirsiniz
        enter_count += 1
    if exit_pixel_count > 50:
        exit_count += 1
    
    # Giriş ve çıkış sayılarını frame üzerine yazdırma
    cv2.putText(frame, f'Enter: {enter_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Exit: {exit_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Sonuçları gösterme
    cv2.imshow('Frame', frame)
    cv2.imshow('Foreground Mask', fgmask)
    
    # 'q' tuşuna basıldığında döngüyü sonlandırma
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırakma
cap.release()
cv2.destroyAllWindows()
