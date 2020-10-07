import cv2
import numpy as np

image = cv2.imread('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/Imagem320x240.jpg')

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Input grayscale image', grayscale_image)

rows, cols = grayscale_image.shape[:2]

output_image = np.zeros((rows, cols), np.uint8)

for row in range(1, rows-1):
    for col in range(1, cols-1):
        gx = grayscale_image[row - 1, col - 1] * (-1) + grayscale_image[row, col - 1] * (-2) + \
             grayscale_image[row + 1, col - 1] * (-1) + grayscale_image[row - 1, col + 1] + \
             grayscale_image[row, col + 1] * 2 + grayscale_image[row + 1, col + 1]

        gy = grayscale_image[row - 1, col - 1] * (-1) + grayscale_image[row - 1, col] * (-2) + \
             grayscale_image[row - 1, col + 1] * (-1) + grayscale_image[row + 1, col - 1] + \
             grayscale_image[row + 1, col] * 2 + grayscale_image[row - 1, col + 1]

        output_image[row, col] = (gx**2 + gy**2)**(1/2)

cv2.imshow('Sobel image', output_image)
cv2.imwrite('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/Imagem320x240.jpg/results/sobel_result.jpg', output_image)
cv2.waitKey(0)