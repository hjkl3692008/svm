import random
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import file_tools as ft
import smo


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
def svm(iteration=1000, c=1, split_rate=1):
    start_time = time.time()
    # load data
    cancer_data = ft.load_cancer()
    # change class to binary code
    last_col = cancer_data.shape[1] - 1
    cancer_data[:, last_col][cancer_data[:, last_col] == 2] = -1
    cancer_data[:, last_col][cancer_data[:, last_col] == 4] = 1

    if split_rate == 1:
        train_data = cancer_data
        test_data = cancer_data
    else:
        train_data, test_data, train_index, test_index = split_train_test(cancer_data, split_rate)

    y = train_data[:, last_col]
    x = train_data[:, 1:last_col]

    w, b = smo.SMO(x, y, iteration=iteration, c=c)
    print('w:'+str(w))
    print('b:'+str(b))
    test_y = test_data[:, last_col]
    test_x = test_data[:, 1:last_col]

    cate_y = svm_category(w, test_x, b)
    fp, fn, tp, tn = fpntpn(test_y, cate_y)
    sen, spec, accuracy = ssa(fp, fn, tp, tn)
    print('fp: %d \nfn: %d \ntp: %d \ntn: %d \n' % (fp, fn, tp, tn))
    print('sensitivity: %1.5f \nspecificity: %1.5f \naccuracy: %1.5f \n' % (sen, spec, accuracy))
    print('SVM execution in ' + str(time.time() - start_time), 'seconds')


# category
def svm_category(w, x, b):
    categories = np.zeros(x.shape[0])
    dim = x.ndim
    if dim == 1:
        categories[0] = category_one(w, x, b)
    else:
        n, d = x.shape
        for i in range(0, n):
            categories[i] = category_one(w, x[i].T, b)
    return categories


# category one data
def category_one(w, x, b):
    wx = np.dot(w, x.T) + b
    cate = sign(wx)
    return cate


# sign
def sign(df):
    return np.sign(df)


# false positive, false negative, true positive, true negative
def fpntpn(actual_v, predict_v):
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    for i in range(0, actual_v.shape[0]):
        if actual_v[i] == 1:
            if actual_v[i] == predict_v[i]:
                tp = tp + 1
            else:
                fn = fn + 1
        if actual_v[i] == -1:
            if actual_v[i] == predict_v[i]:
                tn = tn + 1
            else:
                fp = fp + 1
    return fp, fn, tp, tn


# sensitivity & specificity & accuracy
def ssa(fp, fn, tp, tn):
    if tp + fn == 0:
        sen = 0
    else:
        sen = tp / (tp + fn)
    if tn + fp == 0:
        spec = 0
    else:
        spec = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    return sen, spec, accuracy
