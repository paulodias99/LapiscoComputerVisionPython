import cv2
import numpy as np
import random
from numba import njit

@njit
def region_growing(image):
    rows, cols = image.shape[:2]

    segmented = np.zeros_like(image)

    num_objects = 0

    for ext_row in range(rows):
        for ext_col in range(cols):
            if segmented[ext_row, ext_col] == 0 and image[ext_row, ext_col] < 230:
                num_objects += 1
                segmented[ext_row, ext_col] = num_objects

                current_found = 0
                previous_points = 1

                while previous_points != current_found:
                    previous_points = current_found
                    current_found = 0

                    for row in range(rows):
                        for col in range(cols):
                            if segmented[row, col] == num_objects:
                                if image[row - 1, col - 1] < 230:
                                    segmented[row - 1, col - 1] = num_objects
                                    current_found += 1
                                if image[row - 1, col] < 230:
                                    segmented[row - 1, col] = num_objects
                                    current_found += 1
                                if image[row - 1, col + 1] < 230:
                                    segmented[row - 1, col + 1] = num_objects
                                    current_found += 1
                                if image[row, col - 1] < 230:
                                    segmented[row, col - 1] = num_objects
                                    current_found += 1
                                if image[row, col + 1] < 230:
                                    segmented[row, col + 1] = num_objects
                                    current_found += 1
                                if image[row + 1, col - 1] < 230:
                                    segmented[row + 1, col - 1] = num_objects
                                    current_found += 1
                                if image[row + 1, col] < 230:
                                    segmented[row + 1, col] = num_objects
                                    current_found += 1
                                if image[row + 1, col + 1] < 230:
                                    segmented[row + 1, col + 1] = num_objects
                                    current_found += 1

    return segmented, num_objects


if __name__ == '__main__':
    image = cv2.imread('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/variosobjetos.jpg')
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Original', grayscale_image)

    segmented_image, n_objects = region_growing(grayscale_image)
    rows, cols = segmented_image.shape[:2]
    new_img = np.zeros([rows, cols, 3], np.uint8)

    for n in range(n_objects):
        color = lambda: random.randint(0, 255)

        x_min = 0
        x_max = 0
        y_min = 0
        y_max = 0

        for row in range(rows):
            for col in range(cols):
                if segmented_image[row, col] == n + 1:
                    if x_max < col:
                        x_max = col
                    if x_min > col:
                        x_min = col
                    if y_max < row:
                        y_max = row
                    if y_min > row:
                        y_min = row

                    if x_min == 0:
                        x_min = x_max
                    if y_min == 0:
                        y_min = y_max

        new_img[np.where(segmented_image == n + 1)] = [color(), color(), color()]

        crop_image = new_img[y_min:y_max, x_min:x_max]

        cv2.imshow('Object ' + str(n), crop_image)
        cv2.waitKey(10)

    cv2.waitKey(0)