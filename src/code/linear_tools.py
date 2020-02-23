import operator
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import file_tools as ft


# delete sqft_living >8000
def delete_abnormal(df):
    df = df[df['sqft_living'] < 8000]
    return df


# only contain price and sqft_living
def contain_price_sq(df):
    df = df[['price', 'sqft_living']]
    return df


# plot basic line
def plot_line(data, label=None, title='the basic line'):
    plt.plot(data[:, 0], data[:, 1], 'r--', label=label)
    plt.title(title)
    plt.legend()
    plt.show()


# heat_map  data(dataFrame)
def heat_map(data, title='heat map table'):
    # get basic parameters from data
    col_num = data.shape[1]
    names = data.columns.values
    correction = data.corr()
    # plot correlation matrix
    ax = sns.heatmap(correction, cmap=plt.cm.Greys, linewidths=0.05, vmax=1, vmin=0, annot=True,
                     annot_kws={'size': 6, 'weight': 'bold'})
    plt.xticks(np.arange(col_num) + 0.5, names)
    plt.yticks(np.arange(col_num) + 0.5, names)
    ax.set_title(title)

    plt.show()


# plot scatter points and the linear regression line
def plot_scatter_points(df, w):
    # scatter points
    plt.scatter(df[:, 1], df[:, 0], alpha=0.6)

    # regression line
    axes = plt.gca()
    x = np.array(axes.get_xlim())
    y = w[1] + w[0] * x
    plt.plot(x, y, '--')

    plt.show()


# linear regression
def linear_regression(df, iteration=5000, learn_rate=0.000001):
    assert df.shape[1] == 2

    y = df[:, 0]
    x = df[:, 1]
    # add constant dimension
    x = np.vstack((x, np.ones(x.shape)))

    w = gradient_descent(y, x, learn_rate, iteration)

    return w


# least square method
def gradient_descent(y, x, learn_rate, iteration, is_decay=True, w=np.array([0, 0])):

    for i in range(1, iteration):
        # loss = square_loss(y, x, w)
        dw = square_loss_derivative(y, x, w)
        lr = learn_rate
        if is_decay:
            lr = learn_decay(decay_type=EXPONENTIAL_DECAY, learn_rate=learn_rate, iteration=i)
        w = w - lr * dw

    return w


# square loss
def square_loss(y, x, w):
    loss = np.sum(np.power(np.dot(w.T, x) - y, 2))
    return loss


# least square derivative
def square_loss_derivative(y, x, w):
    dw = np.dot((np.dot(w.T, x) - y), x.T) / x.shape[1]
    return dw

# decay learn rate start ###############


STEP_DECAY = 'STEP_DECAY'
EXPONENTIAL_DECAY = 'EXPONENTIAL_DECAY'
FRACTION_DECAY = 'FRACTION_DECAY'


def learn_decay(decay_type, learn_rate, iteration, par=None):
    lr = None
    if decay_type == STEP_DECAY:
        if par is None:
            lr = step_decay(learn_rate, iteration)
        else:
            lr = step_decay(learn_rate, iteration, n=par)
    elif decay_type == EXPONENTIAL_DECAY:
        if par is None:
            lr = exponential_decay(learn_rate, iteration)
        else:
            lr = exponential_decay(learn_rate, iteration, base=par)
    elif decay_type == FRACTION_DECAY:
        if par is None:
            lr = fraction_decay(learn_rate, iteration)
        else:
            lr = fraction_decay(learn_rate, iteration, rate=par)

    return lr


def step_decay(learn_rate, iteration, n=100):
    times = int(iteration/n)
    lr = learn_rate * np.power(0.5, times)
    return lr


def exponential_decay(learn_rate, iteration, base=0.95):
    lr = learn_rate * np.power(base, iteration)
    return lr


def fraction_decay(learn_rate, iteration, rate=0.5):
    lr = learn_rate / (1 + rate*iteration)
    return lr

# decay learn rate end ###############
