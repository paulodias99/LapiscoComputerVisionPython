import cv2
import matplotlib.pyplot as plt

image = cv2.imread('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/ImagemOriginal.jpg')

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

equalized_image = cv2.equalizeHist(grayscale_image)

original_hist = cv2.calcHist(grayscale_image, channels=[0], mask=None, histSize=[256], ranges=[0, 256])
equalized_hist = cv2.calcHist(equalized_image, channels=[0], mask=None, histSize=[256], ranges=[0, 256])

plt.figure(1)
plt.subplot(221)
plt.imshow(grayscale_image, cmap='gray')
plt.subplot(222)
plt.hist(grayscale_image.ravel(), 256, [0, 256])
plt.subplot(223)
plt.imshow(equalized_image, cmap='gray')
plt.subplot(224)
plt.hist(equalized_image.ravel(), 256, [0, 256])
plt.show()