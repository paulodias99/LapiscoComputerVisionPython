import cv2

#ler
img = cv2.imread("C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/ImagemOriginal.jpg")

#visualizar
cv2.imshow('teste', img)

#salvar

cv2.imwrite("C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/results/imgq1.jpg", img)

cv2.waitKey(0)
