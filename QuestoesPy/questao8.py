import cv2

# Read a rgb image
image = cv2.imread('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/Imagem320x240.jpg')

# Transform to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Input grayscale image', grayscale_image)

rows, cols = grayscale_image.shape[:2]
double_sized_image = cv2.resize(grayscale_image, (2 * rows, 2 * cols))
half_sized_image = cv2.resize(grayscale_image, (int(rows/2), int(cols/2)))

cv2.imshow('Double sized image', double_sized_image)
cv2.imshow('Half sized image', half_sized_image)

cv2.waitKey(0)