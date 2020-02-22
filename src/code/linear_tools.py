import operator
import random
import time

from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures

import file_tools as ft
import time_tools as tt


# split train data and test data
def split_train_test(d, percentage=0.9):
    # num of total data
    shape = d.shape
    num = shape[0]

    # shuffle index
    index = np.arange(num)
    random.shuffle(index)

    # calculate train's num and test's num
    train_num = int(num * percentage)
    test_num = num - train_num

    # get train's index and test's index
    train_index = index[0:train_num]
    test_index = index[train_num:num]

    # split data
    train_data = d[train_index]
    test_data = d[test_index]

    return train_data, test_data, train_index, test_index


class Tdtd:
    test_data = None
    train_data = None

    def __init__(self, test_data, train_data):
        self.test_data = test_data
        self.train_data = train_data


# split train data and test data by fold
def split_by_fold(d, fold):
    num = d.shape[0]
    index = np.arange(num)
    random.shuffle(index)

    d_list = {}
    for i in range(fold):
        i_range = range(0 + i, num, fold)
        train_data = d[index[i_range]]
        d_list[i] = train_data

    tdtd_list = []
    for k1, v1 in d_list.items():
        ov = np.array([])
        for k2, v2 in d_list.items():
            if k2 != k1:
                if ov.size == 0:
                    ov = v2
                else:
                    ov = np.concatenate((ov, v2))
        t = Tdtd(v1, ov)
        tdtd_list.append(t)

    return tdtd_list


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


# minkowski distance
def minkowski(a, b, dim=2):
    return distance.minkowski(a, b, dim)


# false positive, false negative, true positive, true negative
def fpntpn(d, c=None):
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    index = d.shape[1] - 1
    if c is not None:
        d = d[np.where(d[:, index - 1] == c)]
    for i in d:
        if i[index - 1] == 1:
            if i[index] == i[index - 1]:
                tp = tp + 1
            else:
                fn = fn + 1
        if i[index - 1] == 0:
            if i[index] == i[index - 1]:
                tn = tn + 1
            else:
                fp = fp + 1
    return fp, fn, tp, tn


# sensitivity & specificity & accuracy
def ssa(fp, fn, tp, tn):
    sen = tp / (tp + fn)
    spec = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    return sen, spec, accuracy


# normalizing functions start ###############

# todo:// unified entrance
def normalizing(n_type):
    0


# (feature - min)/(max - min)
def mm_normalization(d, contain_label=False):
    assert type(d) == np.ndarray
    n_d = np.array([])
    for i in range(d.shape[1]):
        feature = d[:, i]
        max_f = np.max(feature)
        min_f = np.min(feature)

        n_dc = None
        if contain_label & (i == d.shape[1] - 1):
            n_dc = d[:, i]
        elif max_f == min_f:
            n_dc = d[:, i] / 255  # todo:// 255 not a good choice
        else:
            n_dc = (feature - min_f)/(max_f - min_f)

        if i == 0:
            n_d = n_dc
        else:
            n_d = np.vstack((n_d, n_dc))

    return n_d.T


# (x-μ)/σ
def z_score(d, contain_label=False):
    assert type(d) == np.ndarray
    n_d = np.array([])
    for i in range(d.shape[1]):
        feature = d[:, i]
        mu = np.average(feature)
        std = np.std(feature)

        n_dc = None
        if contain_label & (i == d.shape[1] - 1):
            n_dc = d[:, i]
        elif std == 0:
            n_dc = np.zeros(d[:, i].shape)
        else:
            n_dc = (feature - mu)/std

        if i == 0:
            n_d = n_dc
        else:
            n_d = np.vstack((n_d, n_dc))

    return n_d.T


# 1/(1+sigmoid)
def sigmoid(d, contain_label=False):
    assert type(d) == np.ndarray
    n_d = np.array([])
    for i in range(d.shape[1]):
        if i == 0:
            n_d = 1.0 / (1 + np.exp(-d[:, i]))
        elif contain_label & (i == d.shape[1]-1):
            n_d = np.vstack((n_d, d[:, i]))
        else:
            norm = 1.0 / (1 + np.exp(-d[:, i]))
            n_d = np.vstack((n_d, norm))
    return n_d.T
# normalizing functions end ###############
