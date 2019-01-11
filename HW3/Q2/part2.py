import numpy as np
import pandas as pd
import matplotlib.pyplot as plotter
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


dataset = pd.read_csv('svmdata2.csv')


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
def map_circle(dataset):
    new = np.square(dataset[:, :-1])
    mapp = np.array([1, 1])
    return np.column_stack((np.matmul(new, mapp), dataset[:, -1]))


train_mapped = map_circle(train.values)
test_mapped = map_circle(test.values)

a = train_mapped[:, 0]
b = train_mapped[:, 1]
c = train_mapped[:, 1]
plotter.scatter(a, b, c=c)
plotter.show()

# C
classifier = svm.SVC().fit(train.iloc[:, 0:1], train.iloc[:, 2])
prd = classifier.predict(test.iloc[:, 0:1])
print("Classification accuracy before mapping: ", accuracy_score(prd, test.iloc[:, 2]))

# D
clf = svm.SVC()
clf.fit(train_mapped[:, :-1], train_mapped[:, -1])
prd = clf.predict(test_mapped[:, :-1])
print("Classification accuracy after mapping: ", accuracy_score(prd, test_mapped[:, -1]))
