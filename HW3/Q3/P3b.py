from sklearn.neural_network import MLPClassifier
import pandas as pd

dataset = pd.read_csv("Drinks.csv")
y = dataset[['Class 1', 'Class 2', 'Class 3']].values

# Getting data
X = dataset.drop(['Class 1', 'Class 2', 'Class 3'], axis=1)
X = X.values

y_train = y[0:(int)(0.8 * len(y)), :]
X_train = X[0:(int)(0.8 * len(X)), :]

y_test = y[(int)(0.8 * len(y)):len(y), :]
X_test = X[(int)(0.8 * len(y)):len(X), :]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8, 5, ), activation="tanh", learning_rate_init=0.01, max_iter= 2500)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
