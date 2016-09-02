from sklearn.tree import DecisionTreeRegressor, export_graphviz
import sklearn.cross_validation as cv
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.externals.six import StringIO
import pydotplus
import numpy as np
import pandas as pd
from datetime import datetime
import time
import random
import argparse

DEBUG = False
KFOLDS = 10
MODELS = {
    "dt": DecisionTreeRegressor(min_samples_leaf=8, max_depth=12),
    "et": ExtraTreesRegressor(n_estimators=15),
    "knn": KNeighborsRegressor(),
    "lr": LinearRegression(),
    "svr": SVR()
}

def retrieveDataset(datafile: str):
    data = pd.read_csv(datafile, sep="\t")
    if DEBUG: print(data)

    return data

def getDatasetInfo(data):
    featureLabels = data.columns.values
    print("Labels: ", featureLabels)

    nbFeatures = len(featureLabels)
    print("Number of features: ", nbFeatures)

    nbInstances = len(data)
    print("Number of instances: ", nbInstances)

    return featureLabels, nbFeatures, nbInstances

def getVariablesAndTarget(data, featureLabels):
    X = data[featureLabels[:2]]
    Y = data[featureLabels[2]]

    if DEBUG:
        print("X:")
        print("Length: ", len(X))
        print(X.values)
        print("Y:")
        print("Length: ", len(Y))
        print(Y.values)
    
    return X, Y

def trainTestSplit(X, Y, distribution):
    trainX, testX, trainY, testY = cv.train_test_split(X, Y, test_size=distribution)

    if DEBUG:
        print("Train set X: ")
        print("Length: ", len(trainX))
        print(trainX)
        print("Train set Y: ")
        print("Length: ", len(trainY))
        print(trainY)
        print("Test set X: ")
        print("Length: ", len(testX))
        print(testX)
        print("Test set Y: ")
        print("Length: ", len(testY))
        print(testY)

    return trainX, testX, trainY, testY

def graph(clf, featureLabels):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                        feature_names=featureLabels[:2],
                        class_names=featureLabels[2],
                        filled=True, rounded=True,
                        special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("model.pdf")

def main(datafile: str, algorithm: str):
    data = retrieveDataset(datafile)

    featureLabels, nbFeatures, nbInstances = getDatasetInfo(data)

    #Date to timestamp pre-process for regression
    data[featureLabels[1]] = data[featureLabels[1]].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp())
    if DEBUG: print(data)

    X, Y = getVariablesAndTarget(data, featureLabels)
    trainX, testX, trainY, testY = trainTestSplit(X, Y, 0.3)

    clf = MODELS[algorithm]
    print("Model used: ", clf)
    clf = clf.fit(trainX.values, trainY.values)
    predY = clf.predict(testX)
    if DEBUG: print("Predicted values: ", predY)
    print("Accuracy: ", clf.score(testX.values, testY.values))
    cv_scores = cv.cross_val_score(clf, X.values, Y.values, cv=KFOLDS, scoring="r2")
    print("CV R^2 scores: ", cv_scores)
    print("CV R^2 scores average: {:.2f} (+/- {:.2f})".format(cv_scores.mean(), cv_scores.std() * 2))
    if algorithm in ('et', 'dt'):
        print("Features importance: ", clf.feature_importances_)
    """ need discretisation for the following """
#    print(accuracy_score(testY, predY))
#    print(classification_report(testY, predY, target_names=featureLabels[2]))
#    graph(clf, featureLabels)

def parsing():
  parser = argparse.ArgumentParser(description='Build a model to classify numeric and continuous data, then outputs the model score and predictions')
  parser.add_argument('datafile', nargs=1, help='Path of the data file (example: data.txt)')
  parser.add_argument('algorithm', nargs='?', default='dt', type=str, help='Algorithm (dt (decisiontree), et (extratrees), nb (naïve bayes), knn (knearest_neighbours), lr (linear_regression), svr (support_vector_regression))')
  parser.add_argument('--debug', action='store_const',
                      const=True, default=False,
                      help='Triggers the debug mode with logs (default: false)')

  args = parser.parse_args()
  return vars(args)

if __name__ == '__main__':
  args = parsing()
  DEBUG = args["debug"]
  main(args["datafile"][0], args["algorithm"])
