import cv2

imcolor = cv2.imread('../images/imageq51.jpg')
imlogo = cv2.imread('../images/logoq51.jpg')

imcolor = cv2.cvtColor(imcolor, cv2.COLOR_BGR2GRAY)
imlogo = cv2.cvtColor(imlogo, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()

keypoints_1, descriptors_1 = orb.detectAndCompute(imcolor, None)
keypoints_2, descriptors_2 = orb.detectAndCompute(imlogo, None)

objeto_BFMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = objeto_BFMatcher.match(descriptors_1, descriptors_2)

resultado = cv2.drawMatches(imcolor, keypoints_1, imlogo, keypoints_2, matches[:2], None, flags=2)

cv2.imshow('Resultado', resultado)
cv2.waitKey(0)

# Salvar o resultado
cv2.imwrite('../images/results/sift_result_img.jpg', resultado)