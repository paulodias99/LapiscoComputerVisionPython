import csv
import glob
import os

import cv2


def extract_central_moments(images):
    print('[INFO] Extracting central moments.')
    central_moments = []

    for i, image in enumerate(images):
        print('[INFO] Extracting features of image {}/{}'.format(i + 1, len(images)))
        file = cv2.imread(image)
        file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(file)
        central_moments.append([moments['mu20'], moments['mu11'], moments['mu02'], moments['mu30'],
                                moments['mu21'], moments['mu12'], moments['mu03']])
    print('\n')

    return central_moments

def save_results(extractor_name, features):
    for vector in features:
        print(vector)

    with open(extractor_name + '.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(features)

if __name__ == '__main__':
    dataset = '../dataset/'

    image_paths = glob.glob(os.path.join(dataset, '*.jpg'))
    features = extract_central_moments(image_paths)
    save_results('CentralMoments', features)