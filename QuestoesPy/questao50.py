import cv2
import os
import glob
import csv
import numpy as np
from skimage import feature

def extract_glcm(images, distances, angles):
    print('[INFO] Extracting GLCM.')
    glcm_features = []

    for i, image in enumerate(images):
        print('[INFO] Extracting features of image {}/{}'.format(i + 1, len(images)))
        file = cv2.imread(image)
        file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
        glcm = feature.greycomatrix(file, distances, angles, 256, symmetric=False, normed=True)
        glcm_properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        features = [feature.greycoprops(glcm, glcm_property)[0, 0] for glcm_property in glcm_properties]
        glcm_features.append(features)
    print('\n')
    return glcm_features

def save_results(extractor_name, features):
    for vector in features:
        print(vector)

    with open(extractor_name + '.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(features)

if __name__ == '__main__':
    dataset = '../dataset/'
    image_paths = glob.glob(os.path.join(dataset, '*.jpg'))
    features = extract_glcm(image_paths, distances=[5], angles=[0])
    save_results('GLCM', features)