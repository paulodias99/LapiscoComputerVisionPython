import cv2
import numpy as np

image = cv2.imread('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/variosobjetos.jpg')

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Input grayscale image', grayscale_image)

ret, threshold_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow('Threshold image', threshold_image)

kernel = np.ones((5, 5), np.uint8)

for i in range(7):
    erosion = cv2.erode(threshold_image, kernel, iterations=i)
    # Show the result of the erosion
    cv2.imshow('Dilated image', erosion)
    cv2.waitKey(1000)