import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plotter
from matplotlib import colors
import math


dataset = pd.read_csv('svmdata.csv')

# A
train, test = train_test_split(dataset, test_size=0.3)
train, validation = train_test_split(train, test_size=0.3)
plotter.figure()
a = train.iloc[:, 0]
b = train.iloc[:, 1]
c = train.iloc[:, 2]
plotter.scatter(a, b, c=c)
plotter.colorbar()
plotter.show()


# B
params = [0.1, 1, 10, 100, 1000]
accuracy = []
best_param = -1
for param in params:
    clf = svm.SVC(C=param, gamma='auto')
    a = train.iloc[:, 0:1]
    b = train.iloc[:, 2]
    clf = clf.fit(a, b)
    c = validation.iloc[:, 0:1]
    x = clf.predict(c)
    d = validation.iloc[:, 2]
    tmp = accuracy_score(d, x)
    if tmp > best_param:
        best_param = tmp
    accuracy.append(tmp)

params_labels = ['0.1', '1', '10', '100', '1000']

plotter.scatter(params_labels, accuracy, label='Hyperparameters Accuracy')
plotter.show()


# C
clf = svm.SVC(C=best_param)
train1 = train.values[:, :-1]
train2 = train.values[:, -1]
clf.fit(train1, train2)

class1 = train.values[train2 == -1]
class2 = train.values[train2 == 1]

minC10 = min(class1[:, 0])
maxC10 = max(class1[:, 0])
minC11 = min(class1[:, 1])
maxC11 = max(class1[:, 1])
minC20 = min(class2[:, 0])
maxC20 = max(class2[:, 0])
minC21 = min(class2[:, 1])
maxC21 = max(class2[:, 1])

x_min = min(minC10, minC20) - .5
x_max = max(maxC10, maxC20) + .5
y_min = min(minC11, minC21) - .5
y_max = max(maxC11, maxC21) + .5

clrs = colors.ListedColormap(['red', 'blue'])
boundaries = [-2, 0, 2]
norms = colors.BoundaryNorm(boundaries, clrs.N)

np1 = np.arange(x_min, x_max, .1)
np2 = np.arange(y_min, y_max, .1)
x_new, y_new = np.meshgrid(np1, np2)
z = clf.predict(np.c_[x_new.ravel(), y_new.ravel()])
z = z.reshape(x_new.shape)
plotter.contourf(x_new, y_new, z, cmap=clrs, norm=norms, alpha=0.35)

plotter.scatter(class1[:, 0], class1[:, 1], color='red', label='Class 1')
plotter.scatter(class2[:, 0], class2[:, 1], color='blue', label='Class 2')
plotter.legend()
plotter.show()
