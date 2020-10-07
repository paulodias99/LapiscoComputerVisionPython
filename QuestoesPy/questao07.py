import cv2

image = cv2.imread('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/ImagemOriginal.jpg')

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, threshold_image = cv2.threshold(grayscale_image, 70, 255, cv2.THRESH_BINARY)

cv2.imshow('Input grayscale image', grayscale_image)
cv2.imshow('Threshold result', threshold_image)

cv2.waitKey(0)

