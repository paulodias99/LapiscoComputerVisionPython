import cv2

image = cv2.imread('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/ImagemOriginal.jpg')

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

laplace = cv2.Laplacian(grayscale_image, ddepth=cv2.CV_64F, ksize=3)

laplace = cv2.convertScaleAbs(laplace)

equalized_laplacian = cv2.equalizeHist(laplace)

cv2.imshow('Input grayscale image', grayscale_image)
cv2.imshow('Laplacian filter result', laplace)
cv2.imshow('Equalized Laplacian', equalized_laplacian)

cv2.waitKey(0)