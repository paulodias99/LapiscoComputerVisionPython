import csv
import glob
import os

import cv2
import numpy as np
from skimage import feature


def extract_lbp(images, number_points, radius, eps=1e-7):
    print('[INFO] Extracting LBP.')
    lbp_features = []

    for i, image in enumerate(images):
        print('[INFO] Extracting features of image {}/{}'.format(i + 1, len(images)))
        file = cv2.imread(image)
        file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)

        lbp = feature.local_binary_pattern(file, number_points, radius, method='uniform')

        hist, ret = np.histogram(lbp.ravel(), bins=np.arange(0, number_points + 3), range=(0, number_points + 2))
        hist = hist.astype('float')
        hist /= (hist.sum() + eps)

        image_lbp = [item for item in list(hist)]
        lbp_features.append(image_lbp)
    print('\n')
    return lbp_features

def save_results(extractor_name, features):
    for vector in features:
        print(vector)

    with open(extractor_name + '.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(features)

if __name__ == '__main__':
    dataset = '../dataset/'
    image_paths = glob.glob(os.path.join(dataset, '*.jpg'))
    features = extract_lbp(image_paths, number_points=24, radius=8)
    save_results('LBP', features)