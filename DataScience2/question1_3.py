import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

ChainCode = []
SignalLenght = []
counter = 0
dim = (300,300)

def Recovery_Image(recovery_img, point, chainCode):
    if chainCode == 0:
        recovery_img[point[0] - 1, point[1]] = 255
        return (point[0] - 1, point[1])

    elif chainCode == 1:
        recovery_img[point[0], point[1] + 1] = 255
        return (point[0], point[1] + 1)

    elif chainCode == 2:
        recovery_img[point[0] + 1, point[1]] = 255
        return (point[0] + 1, point[1])

    elif chainCode == 3:
        recovery_img[point[0], point[1] - 1] = 255
        return (point[0], point[1] - 1)

    else:
        print('none')

def verifyNeighborhood(image, point, connectivity): #recebe imagem, ponto de referencia e a conectividade

    global counter
    if connectivity == 4:

        if image[point[0]-1, point[1]] == 255: #verifica se tem algum valor no ponto superior (255 para poder visualizar o numero)
            image[point[0] - 1, point[1]] = 0 #atribui o valor 0 (para visualização)
            ChainCode.append(0)
            SignalLenght.append(counter)
            counter = counter + 1
            return (point[0]-1,point[1])

        elif image[point[0], point[1]+1] == 255:
            image[point[0] , point[1]+1] = 0
            ChainCode.append(1)
            SignalLenght.append(counter)
            counter = counter + 1
            return (point[0],point[1]+1)

        elif image[point[0]+1, point[1]] == 255:
            image[point[0] + 1, point[1]] = 0
            ChainCode.append(2)
            SignalLenght.append(counter)
            counter = counter + 1
            return (point[0]+1,point[1])

        elif image[point[0], point[1]-1] == 255:
            image[point[0], point[1]-1] = 0
            ChainCode.append(3)
            SignalLenght.append(counter)
            counter = counter + 1
            return (point[0],point[1]-1)

        else:
            print('none')
    else:
        print(point)
def normalizeImage(v):
    v = (v - v.min()) / (v.max() - v.min())
    result = (v * 255).astype(np.uint8)
    return result

path='./number1/'

for r, d, f in os.walk(path):

    for filename in f:

        image = cv2.imread(os.path.join(path, filename))
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        imageBin = 255 - image[:, :, 0]

        newImg = np.zeros(np.shape(imageBin))
        kernel = np.ones((3, 3), np.uint8)

        newImg = normalizeImage((imageBin > 100) * 1)  # imagem só tem 0 e 1
        imgCopy = np.copy(newImg)
        imgPlot = np.zeros(np.shape(image))
        imgPlot[:, :, 0] = imgPlot[:, :, 1] = imgPlot[:, :, 2] = imgCopy

        newImg = cv2.dilate(newImg, kernel, iterations=1) - newImg
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        max_xy = np.where(newImg == 255)
        # print(max_xy[0][0] , max_xy[1][0])

        newImgRGB = np.zeros(np.shape(image))
        newImgRGB[:, :, 0] = newImgRGB[:, :, 1] = newImgRGB[:, :, 2] = newImg

        cv2.circle(newImgRGB, (max_xy[1][0], max_xy[0][0]), int(3), (0, 0, 255), 2)

        startPoint = (max_xy[0][0], max_xy[1][0])

        point = verifyNeighborhood(newImg, startPoint, 4)

        while (point != startPoint):
            cv2.circle(imgPlot, (point[1], point[0]), int(3), (0, 0, 255), 4)
            cv2.imshow('image', imgPlot)
            cv2.waitKey(1)

            cv2.circle(imgPlot, (point[1], point[0]), int(3), (0, 255, 255), 6)
            point = verifyNeighborhood(newImg, point, 4)

        recovery_img = np.zeros((500,500))
        current_point = (100, 170)
        for value in ChainCode:
            current_point = Recovery_Image(recovery_img, current_point, value)
            cv2.imshow('recovery image', recovery_img)
            cv2.waitKey(1)

        print('\n======================\nCódigo encontrado:\n', ChainCode, '\n======================')
        ChainCode = []