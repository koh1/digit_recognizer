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

y

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

plot_x_by_y(x, y, colors=['red', 'blue'])
plt.show()


