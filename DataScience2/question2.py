import cv2
import os
import glob
import csv
import numpy as np
from skimage import feature

def extract_hu_moments():
    complete_hu_moments = []

    for i, image in enumerate(path_image):
        print('[INFO] Extracting features of image {}/{}'.format(i + 1, len(self.path_image)))

        file = cv2.imread(image) #lendo a imagem
        file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY) #convertendo para cina

        moments = cv2.moments(file) #fazendo a extração
        hu_moments = cv2.HuMoments(moments) #criando a lista
        new_moments = [moment[0] for moment in hu_moments]
        print(new_moments)

        complete_hu_moments.append(new_moments)

    self.save_results('HUMoments', complete_hu_moments) #salvando o resultado
    return complete_hu_moments

def extract_lbp(number_points, radius, eps=1e-7):
    lbp_features = []

    for i, image in enumerate(self.path_image):
        file = cv2.imread(image)

        file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(file, number_points, radius, method='uniform')

        hist, ret = np.histogram(lbp.ravel(), bins=np.arange(0, number_points + 3), range=(0, number_points + 2))
        hist = hist.astype('float')
        hist /= (hist.sum() + eps)

        image_lbp = [item for item in list(hist)]
        lbp_features.append(image_lbp)

    self.save_results('LBP', lbp_features)
    return lbp_features

def extract_glcm(distances, angles):
    glcm_features = []

    for i, image in enumerate(self.path_image):
        file = cv2.imread(image)

        file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)

        glcm = feature.greycomatrix(file, distances, angles, 256, symmetric=False, normed=True)

        glcm_properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        features = [feature.greycoprops(glcm, glcm_property)[0, 0] for glcm_property in glcm_properties]
        glcm_features.append(features)

    self.save_results('GLCM', glcm_features)
    return glcm_features

def save_results(extractor_name, features):

    for vector in features:
        print(vector)

    with open(extractor_name + '.csv', 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(features)


path = './number1/'

#for para leitura dos arquivos da pasta
for r, d, f in os.walk(path):
    for filename in f:

        extractions = glob.glob(os.path.join(path, filename))
        extractions.extract_hu_moments()
        extractions.extract_lbp(number_points=24, radius=8)
        extractions.extract_glcm(distances=[5], angles=[0])
