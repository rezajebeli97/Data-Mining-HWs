import csv
import matplotlib.pyplot as plotter
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

X = []


def init():
    global X
    with open('iris.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            X.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), row[4]])


def a():
    X_c0 = [row[0] for row in X]
    X_c1 = [row[1] for row in X]
    plotter.hist2d(X_c0, X_c1, 6)
    plotter.colorbar()
    plotter.show()


def b():
    fig = plotter.figure()
    ax = fig.add_subplot(111, projection='3d')
    X_c0 = [row[0] for row in X]
    X_c1 = [row[1] for row in X]
    hist, xedges, yedges = np.histogram2d(X_c0, X_c1, bins=6)
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)
    dx = (xedges[xedges.size - 1] - xedges[0]) / (xedges.size - 1) * np.ones_like(zpos)
    dy = (yedges[yedges.size - 1] - yedges[0]) / (yedges.size - 1) * np.ones_like(zpos)
    dz = hist.flatten()
    offset = dz + np.abs(dz.min())
    fracs = offset.astype(float) / offset.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    color_values = cm.viridis(norm(fracs.tolist()))
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=color_values, zsort='average')
    plotter.show()


def c():
    fig = plotter.figure()
    ax = fig.add_subplot(111, projection='3d')
    X_c0 = [row[0] for row in X]
    X_c1 = [row[1] for row in X]
    X_c2 = [row[2] for row in X]
    ax.scatter(X_c0, X_c1, X_c2, c='black', marker='o')
    ax.set_xlabel('sepal Length')
    ax.set_ylabel('sepal Width')
    ax.set_zlabel('petal Length')
    plotter.show()


def d():
    var = []
    mean = []
    for i in range(len(X[0]) - 1):
        col = [row[i] for row in X]
        var.append(np.var(col))
        mean.append(np.mean(col))
    print('Mean : ', mean)
    print('Variance : ', var)


def e():
    col0 = [row[0] for row in X]
    col1 = [row[1] for row in X]
    cov = np.cov(col0, col1)
    print(cov)


def f():
    col0 = [row[0] for row in X]
    col1 = [row[1] for row in X]
    corr = np.corrcoef(col0, col1)
    print(corr)


def g():
    col0123 = [[row[0], row[1], row[2], row[3]] for row in X if row[-1] == 'Iris-virginica']
    ct = np.transpose(col0123)
    corr = np.corrcoef(ct)
    print(corr)


def h():
    col0123 = [[row[0], row[1], row[2], row[3]] for row in X]
    clrs = {
        'Iris-virginica': 'red',
        'Iris-versicolor': 'yellow',
        'Iris-setosa': 'blue',
    }
    row_colors = [clrs[row[-1]] for row in X]
    sns.clustermap(col0123, row_colors=row_colors, row_cluster=False, col_cluster=False, metric="correlation")
    plotter.show()


init()
h()
