import cv2

image = cv2.imread('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/Imagem320x240.jpg')

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Input grayscale image', grayscale_image)
cv2.waitKey(0)

rows, cols = grayscale_image.shape[:2]

with open('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/results/resultq9.txt', 'w') as outfile:
    for row in range(rows):
        for col in range(cols):
            outfile.write(str(grayscale_image[row, col]) + ' ')
        outfile.write('\n')