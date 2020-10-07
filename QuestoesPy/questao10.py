import cv2
import numpy as np

image = cv2.imread('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/Imagem320x240.jpg')

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Input grayscale image', grayscale_image)
cv2.waitKey(0)

rows, cols = grayscale_image.shape[:2]

threshold_matrix = np.zeros((rows, cols), dtype=np.uint8)

for row in range(rows):
    for col in range(cols):
        threshold_matrix[row, col] = grayscale_image[row, col]

with open('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/results/resultq10.txt', 'w') as outfile:
    for row in range(rows):
        for col in range(cols):
            # Define the limits of the threshold
            if threshold_matrix[row, col] < 127:
                threshold_matrix[row, col] = 0
            else:
                threshold_matrix[row, col] = 255

            outfile.write(str(threshold_matrix[row, col]) + ' ')
        outfile.write('\n')