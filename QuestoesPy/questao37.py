import cv2
image = cv2.imread('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/variosobjetos.jpg')

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Input grayscale image', grayscale_image)

ret, threshold_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow('Threshold image', threshold_image)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

for i in range(9):
    dilation = cv2.dilate(threshold_image, kernel, iterations=i)

    cv2.imshow('Dilated image', dilation)
    cv2.waitKey(1000)