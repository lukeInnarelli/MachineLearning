import csv

from sklearn.neural_network import MLPClassifier
import numpy as np


def learnRisk():
    #gets the dataset
    dataset2 = open('.venv/My model/archive/heart.csv', 'r');
    o = csv.reader(dataset2);
    dataset3 = []
    for dataset2 in o:
        dataset3.append(dataset2);
    dataset4 = np.array(dataset3[1:]);
    dataset5 = np.zeros((len(dataset4),len(dataset4[1])));
    for i in range(0,len(dataset4)):
       for j in range(0,len(dataset4[i])):
            dataset5[i,j] = float(dataset4[i,j]);

    #puts the dataset into the classifier to train
    datasetInput = dataset5[:,:13];
    datasetTarget = dataset5[:,13];
    classifier = MLPClassifier();
    testSize = 10;
    classifier.fit(datasetInput[testSize:] , datasetTarget[testSize:]);
    results = classifier.predict(datasetInput[:testSize]);

    #prints out the results
    print("predicted values");
    print(results);
    print("actual values");
    print(datasetTarget[:testSize]);

    #one line accuracy of the machine learning
    print(np.mean(np.equal(results, datasetTarget[:testSize])));

if __name__ == '__main__':
    learnRisk()