import cv2
import numpy as np

image = cv2.imread('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/variosobjetos.jpg')

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

canny_image = cv2.Canny(grayscale_image, 80, 180)

contours, hierarchy = cv2.findContours(canny_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contour_img = np.copy(image)

cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 3)

cv2.imshow('Input grayscale image', grayscale_image)

cv2.imshow('Contours', contour_img)
cv2.waitKey(0)