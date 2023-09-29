import cv2

# Görüntüleri yükle
image1 = cv2.imread("C:\\Users\\Gulsu\\Desktop\\veri seti\\after37.png", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("C:\\Users\\Gulsu\\Desktop\\veri seti\\before37.png", cv2.IMREAD_GRAYSCALE)

# SIFT özellik dedektörünü oluştur
sift = cv2.SIFT_create()

# Özellik noktalarını ve açıklamalarını bul
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Brute Force Matcher'ı oluştur
bf = cv2.BFMatcher()

# İki görüntü arasında özellik eşleştirmelerini bul
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# İyi eşleşmeleri sakla
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Eşleşen noktaları görselleştir
match_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Eşleşmeleri göster
cv2.imshow("Feature Matches", match_image)
cv2.waitKey(0)
cv2.destroyAllWindows()