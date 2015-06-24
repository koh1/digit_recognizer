import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import matplotlib.animation as animation

names = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species']

iris = pd.read_csv('iris.csv', header=None, names = names)

np.random.seed(1)

FIGSIZE = (5, 3.5)

data = iris[:100]

columns = ['Petal.Width', 'Petal.Length']

x = data[columns]
y = data['Species']

y = (y == 'setosa').astype(int)



def plot_x_by_y(x, y, colors, ax= None):
    if ax is None:
        fig = plt.figure(figsize=FIGSIZE)

        ax = fig.add_subplot(1,1,1)
        fig.subplots_adjust(bottom=0.15)
    x1 = x.columns[0]
    x2 = x.columns[1]

    for (species,group), c in zip(x.groupby(y), colors):
        ax = group.plot(kind='scatter', x = x1, y = x2,
                        color=c, ax = ax, figsize=FIGSIZE)

    return ax

def p_y_given_x(x, w, b):
    def sigmoid(a):
        return 1.0 / (1.0 + np.exp(-a))
    return sigmoid(np.dot(x, w) + b)

def grad(x, y, w, b):
    error = y - p_y_given_x(x, w, b)
    w_grad = -np.mean(x.T * error, axis=1)
    b_grad = -np.mean(error)
    return w_grad, b_grad

def gd(x, y, w, b, eta=0.1, num=100):
    for i in range(1, num):
        w_grad, b_grad = grad(x, y, w, b)
        w -= eta * w_grad
        b -= eta * b_grad
        e = np.mean(np.abs(y - p_y_given_x(x, w, b)))
        yield i, w, b, e

def sgd(x, y, w, b, eta=0.1, num=4):
    for i in range(1, num):
        for index in range(x.shape[0]):
            _x = x.iloc[[index], ]
            _y = y.iloc[[index], ]
            w_grad, b_grad = grad(_x, _y, w, b)
            w -= eta * w_grad
            b -= eta * b_grad
            e = np.mean(np.abs(y - p_y_given_x(x, w, b)))
            yield i, w, b, e

def msgd(x, y, w, b, eta=0.1, num=25, batch_size=10):
    for i in range(1, num):
        for index in range(0, x.shape[0], batch_size):
            _x = x[index:index + batch_size]
            _y = y[index:index + batch_size]
            w_grad, b_grad = grad(_x, _y, w, b)
            w -= eta * w_grad
            b -= eta * b_grad
            e = np.mean(np.abs(y - p_y_given_x(x, w, b)))
            yield i, w, b, e

plot_x_by_y(x, y, colors=['red', 'blue'])
plt.show()


