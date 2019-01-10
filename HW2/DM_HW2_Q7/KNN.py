import numpy as np
import matplotlib.pyplot as plotter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = []
Y = []
x_test = []
y_test = []

def init():  # calculate X, Y
    global X, Y, x_test, y_test
    dataset = pd.read_csv("US Presidential Data.csv")
    X = dataset.iloc[:, 1:].values
    y = dataset.iloc[:, 0].values
    X, x_test, Y, y_test = train_test_split(X, y, test_size=0.25)
    return

def KNN(k):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X, Y)
    predictions = classifier.predict(X)
    return np.linalg.norm(predictions - Y)

def a():
    init()
    error = KNN(1)
    print("error : ", error)

def b():
    init()
    errors = []
    ks = []
    for k in range(1, 50):
        error = KNN(k)
        errors.append(error)
        ks.append(k)
    print(errors)
    plotter.plot(ks, errors)
    plotter.show()

b()
