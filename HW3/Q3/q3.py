import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Preparing the dataset
dataset = pd.read_csv("Drinks.csv")

# Getting the classes
y = dataset[['Class 1', 'Class 2', 'Class 3']].values

# Getting data
X = dataset.drop(['Class 1', 'Class 2', 'Class 3'], axis=1)
X = X.values

# Utility functions


# For the random initialization of the Weights
np.random.seed(0)


def forward_propagation(model, a0):
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']
    W3 = model['W3']
    b3 = model['b3']
    # First ANN layer step
    z1 = a0.dot(W1) + b1
    a1 = np.tanh(z1)
    # Second ANN layer step
    z2 = a1.dot(W2) + b2
    a2 = np.tanh(z2)
    # Third ANN later step
    z3 = a2.dot(W3) + b3
    exp_s = np.exp(z3)
    a3 = exp_s / np.sum(exp_s, axis=1, keepdims=True)

    return {
        "a0": a0,
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "z1": z1,
        "z2": z2,
        "z3": z3
    }

# Found these equations on the internet!
def backward_propagation(model, fp_result, y):
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']
    W3 = model['W3']
    b3 = model['b3']
    a0 = fp_result["a0"]
    a1 = fp_result["a1"]
    a2 = fp_result["a2"]
    a3 = fp_result["a3"]
    num_samples = y.shape[0]

    # Second layer
    dz3 = a3 - y
    dW3 = 1/num_samples*(a2.T).dot(dz3)
    db3 = 1 / num_samples * np.sum(dz3, axis=0)
    # First layer
    dz2 = np.multiply(dz3.dot(W3.T), 1 - np.power(a2, 2))
    dW2 = 1 / num_samples * np.dot(a1.T, dz2)
    db2 = 1 / num_samples * np.sum(dz2, axis=0)
    # Final result
    dz1 = np.multiply(dz2.dot(W2.T), 1 - np.power(a1, 2))
    dW1 = 1 / num_samples * np.dot(a0.T, dz1)
    db1 = 1 / num_samples * np.sum(dz1, axis=0)

    return {
        "dW1": dW1,
        "dW2": dW2,
        "dW3": dW3,
        "db1": db1,
        "db2": db2,
        "db3": db3
    }

# Training the model
def initialize_parameters(nn_input_dim, nn_hdim, nn_output_dim):
    # Weights between 0 and 1
    W1 = 2 * np.random.randn(nn_input_dim, nn_hdim) - 1
    W2 = 2 * np.random.randn(nn_hdim, nn_hdim) - 1
    W3 = 2 * np.random.rand(nn_hdim, nn_output_dim) - 1
    # Bias
    b1 = np.zeros((1, nn_hdim))
    b2 = np.zeros((1, nn_hdim))
    b3 = np.zeros((1, nn_output_dim))

    return {
        "W1": W1,
        "W2": W2,
        "W3": W3,
        "b1": b1,
        "b2": b2,
        "b3": b3,
    }

def update_parameters(model, grads, learning_rate):
    W1, b1, W2, b2, b3, W3 = model['W1'], model['b1'], model['W2'], model['b2'], model['b3'], model["W3"]
    # Updating the weights
    W1 -= learning_rate * grads['dW1']
    W2 -= learning_rate * grads['dW2']
    W3 -= learning_rate * grads['dW3']
    # Updating the biases
    b1 -= learning_rate * grads['db1']
    b2 -= learning_rate * grads['db2']
    b3 -= learning_rate * grads['db3']

    return {
        "W1": W1,
        "W2": W2,
        "W3": W3,
        "b1": b1,
        "b2": b2,
        "b3": b3,
    }

def predict(model, x):
    fp = forward_propagation(model, x)

    return np.argmax(fp['a3'], axis=1)

def accuracy(model, x, y):
    num_examples = y.shape[0]
    prediction = predict(model, x)
    prediction = prediction.reshape(y.shape)
    # By calculating the number of wrongs
    error = np.sum(np.abs(prediction - y))
    return ((num_examples - error) / num_examples) * 100

def train(model, X_, y_, learning_rate, epochs):
    losses = []
    acur = []
    # Epochs iterations
    for i in range(0, epochs):
        fp = forward_propagation(model, X_)
        bp = backward_propagation(model, fp, y_)
        # Updating model parameters
        model = update_parameters(model, bp, learning_rate)
        if (i + 1) % 5 == 0:
            minval = 0.000000000001
            num_samples = y_.shape[0]
            loss = -1 / num_samples * np.sum(y * np.log(fp["a3"].clip(min=minval)))
            print("[epoch=", str(i), "]: Loss=", loss)
            acs = accuracy_score(predict(model, X_), y_.argmax(axis=1))
            print("[epoch=", str(i), "]: acuracy=", acs)
            acur.append(acs)
            losses.append(loss)


    return {
        "model": model,
        "acur": acur,
        "losses": losses
    }

model = initialize_parameters(13, 5, 3)
trained = train(model, X, y, 0.7, 100)
plt.plot(trained["losses"])
plt.show()
plt.plot(trained["acur"])
plt.show()