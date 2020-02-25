import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import file_tools as ft
import smo


# split train data and test data
def split_train_test(d, percentage=0.9, is_save=False):
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

    # save data
    if is_save:
        ft.save_pima(train_data, test_data)

    return train_data, test_data, train_index, test_index


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


# svm
def svm():
    # load data
    cancer_data = ft.load_cancer()
    # change class to binary code
    last_col = cancer_data.shape[1] - 1
    cancer_data[:, last_col][cancer_data[:, last_col] == 2] = -1
    cancer_data[:, last_col][cancer_data[:, last_col] == 4] = 1

    y = cancer_data[:, last_col]
    x = cancer_data[:, 1:last_col]

    w, b = smo.SMO(x, y, iteration=1000, c=1)
    print(w)
    print(b)


# category
def svm_category(w, xs):
    categories = np.zeros(xs.shape[0])
    dim = xs.ndim
    if dim == 1:
        categories[0] = category_one(w, xs)
    else:
        n, d = xs.shape
        for i in range(0, n):
            categories[i] = category_one(w, xs[i].T)


# category one data
def category_one(w, x):
    wx = np.dot(w, x.T)
    cate = sign(wx)
    return cate


# sign
def sign(df):
    return np.sign(df)


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
