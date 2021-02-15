import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.feature import greycomatrix, greycoprops

cap = cv2.VideoCapture('penquinns.mp4')

plt.style.use("ggplot")
(fig, ax) = plt.subplots()
fig.suptitle("Local Binary Patterns")
plt.ylabel("% of Pixels")
plt.xlabel("LBP pixel bucket")
plt.ion()

while (cap.isOpened()):
    ret, frame = cap.read()

    new_width = 300
    new_height = 300

    gray_origin = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    features = local_binary_pattern(gray_origin[:,:,1], 128, 2, method='uniform')
    cv2.imshow("LBP_Image", gray_origin)
    cv2.waitKey(1)

    plt.cla()
    plt.hist(features.ravel(), bins='auto', range=(0, 64))
    plt.pause(0.1)