import cv2

image = cv2.imread('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/variosobjetos.jpg')

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

canny_image = cv2.Canny(grayscale_image, 80, 180)

params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 20
params.maxArea = 40000

params.filterByCircularity = False
params.minCircularity = 0.1

params.filterByConvexity = False
params.minConvexity = 0.87

params.filterByInertia = False
params.minInertiaRatio = 0.8
params.minDistBetweenBlobs = 20

detector = cv2.SimpleBlobDetector_create(params)

blobs = detector.detect(canny_image)

print(len(blobs))

rows, cols = image.shape[:2]

for k in blobs:
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

    cv2.rectangle(image, (x_up_left + 15, y_up_left + 15), (x_bottom_right - 15, y_bottom_right - 15), (255, 0, 0), 2)

cv2.imshow('Result', image)

cv2.imshow('Input grayscale image', grayscale_image)
cv2.waitKey(0)