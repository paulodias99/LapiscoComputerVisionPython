import cv2
import os
import glob
import csv
import numpy as np
from skimage import feature

class Extraction():
    def __init__(self, path):
        self.path = path

    def save_csv(self, extractor_name, features): #função para salvar os resultados em um arquivo csv
        for vector in features:
            print(vector)

        with open(extractor_name + '.csv', 'a') as outfile: #crio arquivo csv
            writer = csv.writer(outfile)
            writer.writerows(features)

    def lbp(self, points, radius, eps=1e-7):
        #print('Iniciando LBP.')
        list_lbp = []

        for i, image in enumerate(self.path):
            arq = cv2.imread(image)

            arq = cv2.cvtColor(arq, cv2.COLOR_BGR2GRAY)

            lbp = feature.local_binary_pattern(arq, points, radius, method='uniform') #extraindo lbp

            hist, ret = np.histogram(lbp.ravel(), bins=np.arange(0, points + 3), range=(0, points + 2)) #calculo do historigrama

            hist = hist.astype('float')
            hist /= (hist.sum() + eps)

            image_lbp = [item for item in list(hist)] #criar um vetor com os features extraídos

            list_lbp.append(image_lbp)

        self.save_csv('LBP', list_lbp)
        return list_lbp

    def hu(self):
        #print('Iniciando HU Moments.')
        list_hu = []

        for i, image in enumerate(self.path):
            arq = cv2.imread(image) #leio a imagem
            arq = cv2.cvtColor(arq, cv2.COLOR_BGR2GRAY) #converto para cinza

            moments = cv2.moments(arq) #faço extração
            HU = cv2.HuMoments(moments) #crio a lista com os features extraídos
            new_hu = [moment[0] for moment in HU]
            print(new_hu)

            list_hu.append(new_hu)

        self.save_csv('HUMoments', list_hu)
        return list_hu

    def glcm(self, distances, angles):
        #print('Iniciando GLCM.')
        list_glcm = []

        for i, image in enumerate(self.path):
            arq = cv2.imread(image)

            arq = cv2.cvtColor(arq, cv2.COLOR_BGR2GRAY)

            glcm = feature.greycomatrix(arq, distances, angles, 256, symmetric=False, normed=True)

            glcm_properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
            features = [feature.greycoprops(glcm, glcm_property)[0, 0] for glcm_property in glcm_properties]

            list_glcm.append(features)

        self.save_csv('GLCM', list_glcm)
        return list_glcm

path = './number1/'

for r, d, f in os.walk(path):
    for filename in f:
        extractions = Extraction(glob.glob(os.path.join(path, filename)))
        extractions.hu()
        extractions.lbp(points=24, radius=8)
        extractions.glcm(distances=[5], angles=[0])
