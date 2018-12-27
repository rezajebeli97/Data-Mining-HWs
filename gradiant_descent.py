import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plotter

X = []
Y = []


def init():  # calculate X, Y
    global X, Y
    a = np.load('data.npz')
    print("Reading data ...")
    for i in range(8000):
        X.append([1, a['x1'][i], a['x2'][i] ** 2, a['x2'][i] ** 2 * a['x1'][i]])
    Y = [[x] for x in a['y']]
    print("Data read!")
    print("Wait for training. It can take a few seconds!")
    return


def f(beta):  # return ||X*beta - Y||^2
    return math.pow(np.linalg.norm(np.subtract(np.matmul(X, beta), Y), 2), 2)


def stoch_f(beta):  # return ||X*beta - Y||^2
    rand = (int)(np.random.rand() * 8000)
    return np.linalg.norm(np.subtract(np.matmul(X[rand], beta), Y), 2) ** 2


def gradiant(beta):
    g = np.subtract(np.matmul(np.matmul(np.transpose(X), X), beta), np.matmul(np.transpose(X), Y))
    g = g / np.linalg.norm(g, 2)
    return g


def stoch_gradiant(beta):
    rand = (int)(np.random.rand() * 8000)
    xrand = [X[rand]]
    yrand = [Y[rand]]
    xt = np.transpose(xrand)
    g = np.subtract(np.matmul(np.matmul(xt, xrand), beta), np.matmul(xt, yrand))
    g = g / np.linalg.norm(g, 2)
    return g


def initialState():
    return [[0], [0], [0], [0]]


def gradiant_descent(maxIteration, alpha):
    beta = initialState()
    f_beta = f(beta)
    iteration = 0
    while f_beta > 0.001 and iteration < maxIteration:
        g = gradiant(beta)
        beta = beta - alpha * g
        f_beta = f(beta)
        iteration += 1
        # print(f_beta)
    print("Beta = ", beta)
    print("SSE = ", f_beta)
    return beta


def stochastic(maxIteration, alpha):
    beta = initialState()
    best_beta = beta
    f_beta = f(beta)
    best_f_beta = f_beta
    iteration = 0
    while f_beta > 0.001 and iteration < maxIteration:
        g = stoch_gradiant(beta)
        beta = beta - alpha * g
        f_beta = f(beta)
        iteration += 1
        # print(f_beta)
        if f_beta < best_f_beta:
            best_beta = beta
            best_f_beta = f_beta
    print("Beta = ", best_beta)
    print("SSE = ", best_f_beta)
    return best_beta


def compare(beta):
    a = np.load('data.npz')
    x_test = []
    for i in range(2000):
        x_test.append([1, a['x1_test'][i], a['x2_test'][i] ** 2, a['x2_test'][i] ** 2 * a['x1_test'][i]])
    x1_test = [[i] for i in a['x1_test']]
    x2_test = [[i] for i in a['x2_test']]
    y_test = [[i] for i in a['y_test']]
    y_predict = np.matmul(x_test, beta)
    sse_test = np.linalg.norm(np.subtract(y_test, y_predict), 2) ** 2
    sse_train = f(beta)
    print("SSE for train data : ", sse_train)
    print("SSE for test data : ", sse_test)
    fig = plotter.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x1_test, x2_test, y_test, color='red')
    ax.scatter(x1_test, x2_test, y_predict, color='blue')
    plotter.show()


init()
beta = gradiant_descent(1000, 0.01)
# beta = stochastic(5000, 0.01)
compare(beta=beta)
