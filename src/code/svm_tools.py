import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import file_tools as ft


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


# todo:// how to complement data


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


# fpntpn digit from 0~9
def fpntpn_digit(d):
    c_range = range(0, 10)
    fpntpn_list = np.array([])
    for c in c_range:
        fp, fn, tp, tn = fpntpn(d, c)
        ftpn = np.array([fp, fn, tp, tn])
        if c == 0:
            fpntpn_list = ftpn
        else:
            fpntpn_list = np.hstack((fpntpn_list, ftpn))
    return fpntpn_list


# sensitivity & specificity & accuracy
def ssa(fp, fn, tp, tn):
    sen = tp / (tp + fn)
    spec = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    return sen, spec, accuracy
