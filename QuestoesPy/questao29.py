import cv2
import numpy as np

image = cv2.imread('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/variosobjetos.jpg')

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(grayscale_image, cv2.HOUGH_GRADIENT, 1, 30,
                           param1=150, param2=25, minRadius=0, maxRadius=0)

try:
    circles = np.uint16(np.around(circles))
except AttributeError:
    print('None circles found! Try change the parameters.')
    exit()

circles_img = np.copy(image)

for xc, yc, radius in circles[0, :]:
    cv2.circle(circles_img, (xc, yc), radius, (0, 0, 255), 2)

cv2.imshow('Input grayscale image', grayscale_image)

cv2.imshow('Threshold result', circles_img)
cv2.waitKey(0)