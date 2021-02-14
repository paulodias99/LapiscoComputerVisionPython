import cv2
import numpy as np
import matplotlib as plt

ChainCode = []
SignalLenght = []
counter = 0
dim = (300,300)

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

image = cv2.imread("C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/DataScience2/number1/1_5.jpg")
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
imageBin = 255 - image[:,:,0] #converter em img binaria

newImg = np.zeros(np.shape(imageBin))
kernel = np.ones((3,3), np.uint8)

newImg = normalizeImage((imageBin>100)*1) #imagem só tem 0 e 1
imgCopy = np.copy(newImg)
imgPlot = np.zeros(np.shape(image))
imgPlot[:,:,0] = imgPlot[:,:,1] = imgPlot[:,:,2] = imgCopy

newImg = cv2.dilate(newImg, kernel, iterations = 1) - newImg # dilatar imagem e subtrair da imagem dilatada a imagem original. isso resultará apenas na borda

max_xy = np.where(newImg == 255)
#print(max_xy[0][0] , max_xy[1][0])

newImgRGB = np.zeros(np.shape(image))
newImgRGB[:,:,0] = newImgRGB[:,:,1] = newImgRGB[:,:,2] = newImg

cv2.circle(newImgRGB, (max_xy[1][0],max_xy[0][0]) , int(3), (0,0,255), 2)

startPoint = (max_xy[0][0],max_xy[1][0])

point = verifyNeighborhood(newImg, startPoint, 4)

while(point != startPoint):

    cv2.circle(imgPlot, (point[1], point[0]), int(3), (0, 0, 255), 4)
    cv2.imshow('image', imgPlot)
    cv2.waitKey(1)

    cv2.circle(imgPlot, (point[1], point[0]), int(3), (0, 255, 255), 6)
    point = verifyNeighborhood(newImg, point, 4)

print('\n======================\nCódigo encontrado: \n',ChainCode,'\n======================')
cv2.imshow('image', imgPlot)
cv2.waitKey(0)