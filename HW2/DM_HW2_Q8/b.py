import string
import numpy as np
import pandas
import graphviz
from sklearn import tree
from sklearn import preprocessing
from matplotlib import pyplot as plotter

dataset = pandas.read_csv("noisy_train.csv")
dataset_test = pandas.read_csv("noisy_test.csv")
dataset_valid = pandas.read_csv("noisy_valid.csv")

X = dataset.iloc[:, 1:].values
Y = dataset.iloc[:, 0].values

x_test = dataset_test.iloc[:, 1:].values
y_test = dataset_test.iloc[:, 0].values

x_valid = dataset_valid.iloc[:, 1:].values
y_valid = dataset_valid.iloc[:, 0].values

prep = preprocessing.LabelEncoder()
prep.fit([x for x in string.ascii_lowercase])

X_encoded = []
for row in X:
    X_encoded.append(prep.transform(row))  # x_encoded has transformed x values from string to float

x_valid_encoded = []
for row in x_valid:
    x_valid_encoded.append(prep.transform(row))  # x_valid_encoded has transformed x_valid values from string to float

x_test_encoded = []
for row in x_test:
    x_test_encoded.append(prep.transform(row))  # x_test_encoded has transformed x_test values from string to float

nodes = []
train_errors = []
test_errors = []
valid_errors = []

for i in range(1, 30):
    classifier = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=5, max_depth=i).fit(X_encoded, Y)

    dot_data = tree.export_graphviz(classifier, out_file=None,
                                    feature_names=dataset.columns[1:],
                                    class_names=['poisonous', 'non-poisonous'],
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("graph/Q8B" + str(i) + "_" + str(classifier.tree_.node_count))

    y_pred = classifier.predict(X_encoded)
    train_error = np.linalg.norm(Y - y_pred)
    print("Euclidean error for train data : ", train_error)

    y_valid_pred = classifier.predict(x_valid_encoded)
    valid_error = np.linalg.norm(y_valid - y_valid_pred)
    print("Euclidean error for valid data : ", valid_error)

    y_test_pred = classifier.predict(x_test_encoded)
    test_error = np.linalg.norm(y_test - y_test_pred)
    print("Euclidean error for test data : ", test_error ,"\n")

    nodes.append(classifier.tree_.node_count)
    train_errors.append(train_error)
    valid_errors.append(valid_error)
    test_errors.append(test_error)

plotter.figure(0)
plotter.plot(nodes, train_errors)
plotter.figure(1)
plotter.plot(nodes, valid_errors)
plotter.figure(2)
plotter.plot(nodes, test_errors)

plotter.show()
