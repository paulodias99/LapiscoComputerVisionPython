import numpy as np
import math
import pandas as pd
import csv

def hold_out(df, train_size, shuffle=True):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    data = []
    for row in df.iterrows():
        index, values = row
        data.append(values.tolist())

    X_train = data[:int(train_size*len(data))]
    X_test = data[int(train_size*len(data)):]

    y_train = [int(x[-1]) for x in X_train]
    y_test = [int(x[-1]) for x in X_test]

    X_train = [x[:-1] for x in X_train]
    X_test = [x[:-1] for x in X_test]

    return X_train, X_test, y_train, y_test

def leave_one_out(df, shuffle=True):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    data = []
    for row in df.iterrows():
        index, values = row
        data.append(values.tolist())

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for i in range(len(data)):
        train = data.copy()
        train.remove(data[i])

        test = data[i]

        y_train.append([int(x[-1]) for x in train])
        y_test.append(int(test[-1]))

        X_train.append([x[:-1] for x in train])
        X_test.append(test[:-1])

    return X_train, X_test, y_train, y_test


def read_data(file):
    return pd.read_csv(file, sep=',', header=None)

def euclidean_dist(x1, x2):
    dist = 0.0
    for x, y in zip(x1, x2):
        dist += pow(float(x) - float(y), 2)
    dist = math.sqrt(dist)

    return dist

class KMeans:
    def __init__(self, k=3, tolerance=0.0001, max_iterations=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.centroids = {}

    def fit(self, data):
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iterations):
            self.classes = {}
            for i in range(self.k):
                self.classes[i] = []

            for features in data:
                distances = [np.linalg.norm(np.array(features) - np.array(self.centroids[centroid])) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)

            previous = dict(self.centroids)

            for classification in self.classes:
                self.centroids[classification] = np.average(self.classes[classification], axis=0)

            isOptimal = True

            for centroid in self.centroids:
                original_centroid = previous[centroid]

                current = self.centroids[centroid]

                if np.sum((current - original_centroid)/original_centroid * 100) > self.tolerance:
                    isOptimal = False

            if isOptimal:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

if __name__ == '__main__':
    filename = 'C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/features.txt'
    features = read_data(filename)

    X_train, X_test, y_train, y_test = hold_out(features, train_size=0.9)

    km = KMeans(k=3, max_iterations=10000)
    km.fit(X_train)

    predictions = []
    for test in X_test:
        predictions.append(km.predict(test))

    count = 0
    for x, y in zip(y_test, predictions):
        if x == y:
            count += 1

    accuracy = count/len(y_test)
    print('Accuracy using hold out: {:.4f}'.format(accuracy))

    with open('C:/Users/User/Desktop/GIT/LapiscoComputerVisionPython/images/results/true_and_predict_54.csv', 'w') as outfile:
        rows = [y_test, predictions]
        writer = csv.writer(outfile, delimiter=',')
        writer.writerows(rows)

    X_train, X_test, y_train, y_test = leave_one_out(features)
    count = 0
    for train_set, test_set, label_train, label_test in zip(X_train, X_test, y_train, y_test):
        km = KMeans(k=3, max_iterations=10000)
        km.fit(train_set)

        predictions = []

        predict = km.predict(test_set)

        if predict == label_test:
            count += 1

    accuracy = count / len(y_test)
    print('Accuracy using leave one out: {:.4f}'.format(accuracy))