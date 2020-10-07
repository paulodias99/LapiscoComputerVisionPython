import cv2

image = cv2.imread('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/ImagemOriginal.jpg')

# Canais
blue_channel, green_channel, red_channel = cv2.split(image)

cv2.imshow('Blue Channel', blue_channel)
cv2.imshow('Green Channel', green_channel)
cv2.imshow('Red Channel', red_channel)

cv2.imwrite('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/results/blue_channel.jpg', blue_channel)
cv2.imwrite('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/results/green_channel.jpg', green_channel)
cv2.imwrite('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/results/red_channel.jpg', red_channel)

cv2.waitKey(0)