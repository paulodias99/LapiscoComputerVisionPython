import cv2
import numpy as np

def verifyNeighborhood(image, point, connectivity): #recebe imagem, ponto de referencia e a conectividade

    if connectivity == 4:
        print(point)
        if image[point[0]-1, point[1]] == 255: #verifica se tem algum valor no ponto superior (255 para poder visualizar o numero)
            image[point[0] - 1, point[1]] = 0 #atribui o valor 0 (para visualização)
            print('0')
            return (point[0]-1,point[1])

        elif image[point[0], point[1]+1] == 255:
            image[point[0] , point[1]+1] = 0
            print('1') #caso vá para a direita, atribui 0 na linha acima e printa 1
            return (point[0],point[1]+1)

        elif image[point[0]+1, point[1]] == 255:
            image[point[0] + 1, point[1]] = 0
            print('2')
            return (point[0]+1,point[1])

        elif image[point[0], point[1]-1] == 255:
            image[point[0], point[1]-1] = 0
            print('3')
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
#visualizar
imageBin = 255 - image[:,:,0] #converter em img binaria

newImg = np.zeros(np.shape(imageBin))

newImg = normalizeImage((imageBin>100)*1) #imagem só tem 0 e 1
max_xy = np.where(newImg == 255)
print(max_xy[0][0] , max_xy[1][0])

newImgRGB = np.zeros(np.shape(image))
newImgRGB[:,:,0] = newImgRGB[:,:,1] = newImgRGB[:,:,2] = newImg

cv2.circle(newImgRGB, (max_xy[1][0],max_xy[0][0]) , int(3), (0,0,255), 2)

startPoint = (max_xy[0][0],max_xy[1][0])

point = verifyNeighborhood(newImg, startPoint, 4)

while(point != startPoint):

    cv2.circle(newImgRGB, (point[1], point[0]), int(7), (0, 0, 255), 1)
    cv2.imshow('image', newImgRGB)
    cv2.waitKey(0)

    cv2.circle(newImgRGB, (point[1], point[0]), int(6), (0, 255, 255), 1)
    point = verifyNeighborhood(newImg, point, 4)
