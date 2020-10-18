import csv
import glob
import os

import cv2


def extract_hu_moments(images):
    print('[INFO] Extracting HU Moments.')
    complete_hu_moments = []

    for i, image in enumerate(images):
        print('[INFO] Extracting features of image {}/{}'.format(i + 1, len(images)))
        file = cv2.imread(image)
        file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)

        moments = cv2.moments(file)

        hu_moments = cv2.HuMoments(moments)
        new_moments = [moment[0] for moment in hu_moments]
        complete_hu_moments.append(new_moments)

    print('\n')
    return complete_hu_moments

def save_results(extractor_name, features):
    for vector in features:
        print(vector)

    with open(extractor_name + '.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(features)

if __name__ == '__main__':
    dataset = '../dataset/'
    image_paths = glob.glob(os.path.join(dataset, '*.jpg'))
    features = extract_hu_moments(image_paths)
    save_results('HUMoments', features)
