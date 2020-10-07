import cv2

image = cv2.imread('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/ImagemOriginal.jpg')

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_image)

cv2.imshow('HSV Image', hsv_image)
cv2.imshow('H Channel', h)
cv2.imshow('S Channel', s)
cv2.imshow('V Channel', v)

cv2.imwrite('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/results/hsv_image.jpg', hsv_image)
cv2.imwrite('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/results/h_channel.jpg', h)
cv2.imwrite('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/results/s_channel.jpg', s)
cv2.imwrite('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/results/v_channel.jpg', v)

cv2.waitKey(0)
