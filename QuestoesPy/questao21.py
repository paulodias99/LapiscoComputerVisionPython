import cv2
import numpy as np
from numba import njit

seed = (0, 0)

@njit
def region_growing(image, seed=None):
    rows, cols = image.shape[:2]
    xc, yc = seed
    segmented = np.zeros_like(image)
    segmented[xc, yc] = 255

    current_found = 0
    previous_points = 1

    while previous_points != current_found:

        previous_points = current_found
        current_found = 0
        for row in range(rows):
            for col in range(cols):
                if segmented[row, col] == 255:
                    if image[row - 1, col - 1] < 127:
                        segmented[row - 1, col - 1] = 255
                        current_found += 1
                    if image[row - 1, col] < 127:
                        segmented[row - 1, col] = 255
                        current_found += 1
                    if image[row - 1, col + 1] < 127:
                        segmented[row - 1, col + 1] = 255
                        current_found += 1
                    if image[row, col - 1] < 127:
                        segmented[row, col - 1] = 255
                        current_found += 1
                    if image[row, col + 1] < 127:
                        segmented[row, col + 1] = 255
                        current_found += 1
                    if image[row + 1, col - 1] < 127:
                        segmented[row + 1, col - 1] = 255
                        current_found += 1
                    if image[row + 1, col] < 127:
                        segmented[row + 1, col] = 255
                        current_found += 1
                    if image[row + 1, col + 1] < 127:
                        segmented[row + 1, col + 1] = 255
                        current_found += 1

    return segmented


def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global seed

        seed = (y, x)


if __name__ == '__main__':
    image = cv2.imread('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/imagem2.jpg')

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('Original Image', 1)
    cv2.imshow('Original Image', grayscale_image)
    cv2.setMouseCallback('Original Image', mouse_event)
    cv2.waitKey(0)

    segmented_image = region_growing(grayscale_image, seed)

    cv2.imshow('Segmented image', segmented_image)
    cv2.waitKey(0)