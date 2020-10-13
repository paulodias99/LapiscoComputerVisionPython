import cv2

image = cv2.imread('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/variosobjetos.jpg')

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

canny_image = cv2.Canny(grayscale_image, 80, 180)

params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 20
params.maxArea = 10000

params.filterByCircularity = False
params.minCircularity = 0.1

params.filterByConvexity = True
params.minConvexity = 0.7

params.filterByInertia = True
params.minInertiaRatio = 0.8

params.minDistBetweenBlobs = 20

# Create a blob detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

blobs = detector.detect(canny_image)

print(len(blobs))

rows, cols = image.shape[:2]

for i, k in enumerate(blobs):
    x_up_left = int(k.pt[0] - k.size)
    y_up_left = int(k.pt[1] - k.size)

    x_bottom_right = int(k.pt[0] + k.size)
    y_bottom_right = int(k.pt[1] + k.size)

    if x_up_left < 0:
        x_up_left = 0
    if y_up_left < 0:
        y_up_left = 0

    if x_bottom_right > cols:
        x_bottom_right = cols
    if y_bottom_right > rows:
        y_bottom_right = rows

    crop = image[y_up_left + 15:y_bottom_right - 15, x_up_left + 15:x_bottom_right - 15]

    cv2.imshow('Object ' + str(i + 1), crop)
    cv2.waitKey(10)


cv2.imshow('Input grayscale image', grayscale_image)
cv2.waitKey(0)