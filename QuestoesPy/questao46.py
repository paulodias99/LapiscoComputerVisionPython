import cv2
import os
import glob
import csv

def extract_spatial_moments(images):
    print('[INFO] Extracting spatial moments.')
    spatial_moments = []

    for i, image in enumerate(images):
        print('[INFO] Extracting features of image {}/{}'.format(i + 1, len(images)))
        file = cv2.imread(image)
        file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(file)
        spatial_moments.append([moments['m00'], moments['m10'], moments['m01'], moments['m20'], moments['m11'],
                                moments['m02'], moments['m30'], moments['m21'], moments['m12'], moments['m03']])

    print('\n')
    return spatial_moments


def save_results(extractor_name, features):
    for vector in features:
        print(vector)

    with open(extractor_name + '.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(features)

if __name__ == '__main__':
    dataset = '../dataset'

    image_paths = glob.glob(os.path.join(dataset, '*.jpg'))

    features = extract_spatial_moments(image_paths)

    save_results('SpatialMoments', features)