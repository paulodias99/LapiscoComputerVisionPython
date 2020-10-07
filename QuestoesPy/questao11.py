import cv2
import numpy as np

image = cv2.imread('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/ImagemT11.jpg')

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Input grayscale image', grayscale_image)

rows, cols = grayscale_image.shape[:2]

new_image = np.zeros((rows, cols), dtype=np.uint8)

for row in range(rows):
    for col in range(cols):
        new_image[row, col] = grayscale_image[row, col]

xc = 0
yc = 0
count = 0

for row in range(rows):
    for col in range(cols):
        if new_image[row, col] == 0:
            xc += row
            yc += col
            count += 1

xc = int(xc/count)
yc = int(yc/count)

cv2.circle(new_image, (xc, yc), 5, (255, 255, 255), -1)

cv2.imshow('Centroid', new_image)
cv2.waitKey(0)

cv2.imwrite('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/results/centroid.jpg', new_image)