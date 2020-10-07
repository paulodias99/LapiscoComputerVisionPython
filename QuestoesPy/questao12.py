import cv2
import numpy as np

filename = 'C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/results/resultq10.txt'
image = []
with open(filename, 'r') as infile:
    for i, line in enumerate(infile):
        row = [int(number) for number in line.split()]
        if i == 0:
            image = np.hstack(row)
        else:
            image = np.vstack(([image, row]))

result = np.asarray(image, np.uint8)

cv2.imshow('Read image', result)
cv2.waitKey(0)