import cv2

image = cv2.imread('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/ImagemOriginal.jpg')

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

median_image = cv2.medianBlur(grayscale_image, ksize=5)
blur_image = cv2.blur(grayscale_image, ksize=(5, 5))

cv2.imshow('Input grayscale image', grayscale_image)

cv2.imshow('Median filter result', median_image)
cv2.imshow('Blur filter result', blur_image)

cv2.imwrite('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/results/median_filter_result.jpg', median_image)
cv2.imwrite('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/results/blur_filter_result.jpg', blur_image)

cv2.waitKey(0)