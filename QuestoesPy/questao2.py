import cv2

#ler
img = cv2.imread("C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/ImagemOriginal.jpg")
cv2.imshow('imagem original', img)

#converter em tom cinza
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('imagem cinza', img_gray)

#salvar imagem gerada
cv2.imwrite("C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/results/imgq2.jpg", img_gray)

cv2.waitKey(0)
