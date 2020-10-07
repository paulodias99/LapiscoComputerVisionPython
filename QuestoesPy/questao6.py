import cv2

image = cv2.imread('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/ImagemOriginal.jpg')

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny_image = cv2.Canny(grayscale_image, 80, 180)

cv2.imshow('Input grayscale image', grayscale_image)
cv2.imshow('Canny filter result', canny_image)

cv2.imwrite('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/results/canny_filter_result.jpg', canny_image)
cv2.waitKey(0)

