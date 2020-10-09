import cv2
import numpy as np
from numba import njit

seed = (0, 0)

#@njit
def region_growing(image, seed=None):
    rows, cols = image.shape[:2]
    xc, yc = seed
    ref_color = image[xc, yc]

    segmented = np.zeros_like(image)

    segmented[xc, yc] = ref_color

    current_found = 0
    previous_points = 1

    while previous_points != current_found:

        previous_points = current_found
        current_found = 0
        for row in range(rows):
            for col in range(cols):
                if np.array_equal(segmented[row, col], ref_color):
                    if np.array_equal(image[row - 1, col - 1], ref_color):
                        segmented[row - 1, col - 1] = ref_color
                        current_found += 1
                    if np.array_equal(image[row - 1, col], ref_color):
                        segmented[row - 1, col] = ref_color
                        current_found += 1
                    if np.array_equal(image[row - 1, col + 1], ref_color):
                        segmented[row - 1, col + 1] = ref_color
                        current_found += 1
                    if np.array_equal(image[row, col - 1], ref_color):
                        segmented[row, col - 1] = ref_color
                        current_found += 1
                    if np.array_equal(image[row, col + 1], ref_color):
                        segmented[row, col + 1] = ref_color
                        current_found += 1
                    if np.array_equal(image[row + 1, col - 1], ref_color):
                        segmented[row + 1, col - 1] = ref_color
                        current_found += 1
                    if np.array_equal(image[row + 1, col], ref_color):
                        segmented[row + 1, col] = ref_color
                        current_found += 1
                    if np.array_equal(image[row + 1, col + 1], ref_color):
                        segmented[row + 1, col + 1] = ref_color
                        current_found += 1

        cv2.imshow('Segmentation', segmented)
        cv2.waitKey(1)

    return segmented


def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global seed

        seed = (y, x)


if __name__ == '__main__':
    image = cv2.imread('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/ImagemOriginal.jpg')
    image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)

    cv2.namedWindow('Original Image', 1)
    cv2.imshow('Original Image', image)
    cv2.setMouseCallback('Original Image', mouse_event)
    cv2.waitKey(0)

    segmented_image = region_growing(image, seed)

    cv2.imshow('Segmented image', segmented_image)
    cv2.waitKey(0)
