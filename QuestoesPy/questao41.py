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

cv2.imshow('Input grayscale image', grayscale_image)
cv2.waitKey(0)