import numpy as np
import random


# SMO
def SMO(x, y, iteration=1000, c=1):
    w, b, a = init_wba(x)
    e, u = update_e(w, x, b, y)

    i = 0
    while i < iteration:
        a_changed = 0
        for j in range(0, x.shape[0]):
            i = i + 1
            a1_old = find_a1(a, y, u, c)
            a2_old = find_a2(w, x, b, y, u, c, a, a1_old)
            a1_new, a2_new = update_Ls(w, x, b, y, c, a, a1_old, a2_old)
            if a[a1_old] != a1_new or a[a2_old] != a2_new:
                a_changed = a_changed + 1
            a1_old_v = a[a1_old]
            a2_old_v = a[a2_old]
            a[a1_old] = a1_new
            a[a2_old] = a2_new
            w = update_w(x, y, a)
            # b = update_b(a, y, w, x)
            e, u = update_e(w, x, b, y)
            b = update_b(x, b, y, e, a1_old, a2_old, a1_old_v, a2_old_v, a1_new, a2_new)
        # if a never changes in a entire loop, break
        if a_changed == 0:
            break
    print('total iteration:'+str(i))
    return w, b


def init_wba(x):
    w = np.zeros(x.shape[1])
    b = 0
    a = np.ones(x.shape[0])
    return w, b, a


# update w
def update_w(x, y, a):
    ys = np.tile(y, (x.shape[1], 1)).T
    xc = np.multiply(x, ys)
    w = np.dot(a, xc)
    return w


# return index of a which >0
def find_great_than_zero_a(a):
    for i in range(0, a.shape[0]):
        if a[i] > 0:
            return i


# update b by ais, c, w, x (only need one point which ai>0)
# def update_b(a, y, w, x):
#     index = find_great_than_zero_a(a)
#     yi = y[index]
#     xi = x[index]
#     b = yi - np.dot(w, xi)
#     return b
def update_b(x, b, y, e, a1, a2, a1_old, a2_old, a1_new, a2_new):
    x1 = x[a1]
    x2 = x[a2]
    y1 = y[a1]
    y2 = y[a2]
    e1 = e[a1]
    b = b - e1 - y1 * np.dot(x1, x1) * (a1_new - a1_old) - y2 * np.dot(x1, x2) * (a2_new - a2_old)
    return b


# whether meet KTT condition
def ktt(ai, yi, ui, c):
    is_meet = False
    if ai == 0 and yi * ui >= 1:  # normal category inner
        is_meet = True
    elif 0 < ai < c and yi * ui == 1:  # support vector
        is_meet = True
    elif ai == c and yi * ui <= 1:  # abnormal category outer
        is_meet = True
    return is_meet


# find a1 which does not meet ktt condition
def find_a1(a, y, u, c):
    for i in range(0, a.shape[0]):
        ai = a[i]
        yi = y[i]
        ui = u[i]
        is_meet = ktt(ai, yi, ui, c)
        if not is_meet:
            return i

    rand = random_num(-1, y.shape[0])
    return rand


# find a2, in which |e1 - e2|max
def find_a2(w, x, b, y, u, c, a, a1):
    # calculate e1
    x1 = x[a1]
    y1 = y[a1]
    e1 = Ei(w, x1, b, y1)

    # find a2 indexes
    l_meet = l_meet_ktt(a, y, u, c)

    # if all data do not meet ktt, return a random num(!=a1)
    if len(l_meet) == 0:
        a2 = random_num(a1, x.shape[0])
        return a2

    max_index = 0
    max_value = 0
    for i in l_meet:
        if i == a1:
            continue
        e2 = Ei(w, x[i], b, y[i])
        diff = np.abs(e1 - e2)
        if diff > max_value:
            max_value = diff
            max_index = i

    return max_index


# l_meet_ktt
def l_meet_ktt(a, y, u, c):
    l_meet = []
    for i in range(0, a.shape[0]):
        ai = a[i]
        yi = y[i]
        ui = u[i]
        is_meet = ktt(ai, yi, ui, c)
        if is_meet:
            l_meet.append(i)
    l_meet = np.array(l_meet)
    return l_meet


# eta
def eta(x1, x2):
    eta_v = 2 * dot(x1, x2) - dot(x1, x1) - dot(x2, x2)
    return eta_v


# predictive value
def Ui(w, xi, b):
    ui = dot(w, xi) + b
    return ui


# error
def Ei(w, xi, b, yi):
    ui = Ui(w, xi, b)
    e = ui - yi
    return e


def update_Ls(w, x, b, y, c, a, a1, a2):
    a1_old = a[a1]
    a2_old = a[a2]
    x1 = x[a1]
    x2 = x[a2]
    y1 = y[a1]
    y2 = y[a2]
    eta_v = eta(x1, x2)
    e1 = Ei(w, x1, b, y1)
    e2 = Ei(w, x2, b, y2)
    low_b, high_b = upper_lower_bound(c, a1_old, a2_old, y1, y2)
    a2_new = update_a2(a2_old, y2, e1, e2, eta_v, low_b, high_b)
    a1_new = update_a1(a1_old, a2_old, a2_new, y1, y2)
    return a1_new, a2_new


# find upper and lower bound of a
def upper_lower_bound(c, a1_old, a2_old, y1, y2):
    if y1 * y2 < 0:
        L = np.maximum(0, a2_old - a1_old)
        H = np.minimum(c, c + a2_old - a1_old)
    else:
        L = np.maximum(0, a2_old + a1_old -c)
        H = np.minimum(c, a1_old + a2_old)

    return L, H


def update_a2(a2_old, y2, e1, e2, eta_v, low_b, high_b):
    a2_new = a2_old - y2 * (e1 - e2) / eta_v
    if a2_new > high_b:
        a2_new = high_b
    elif a2_new < low_b:
        a2_new = low_b
    return a2_new


# return 2 array, e_array and u_array
def update_e(w, x, b, y):
    us = []
    es = []
    for i in range(0, x.shape[0]):
        xi = x[i]
        yi = y[i]
        ui = Ui(w, xi, b)
        us.append(ui)
        ei = ui - yi
        es.append(ei)

    es = np.array(es)
    return es, us


def update_a1(a1_old, a2_old, a2_new, y1, y2):
    a1_new = a1_old + y1 * y2 * (a2_old - a2_new)
    return a1_new


# produce a random num from 0 to length-1 which does not equal to i
def random_num(i, length):
    num = i
    while num == i:
        num = random.randint(0, length-1)
    return num


# np.dot
def dot(x1, x2):
    assert len(x1) == len(x2)
    k = np.dot(x1, x2)
    return k


# np.multiply
def mul(x1, x2):
    m = np.multiply(x1, x2)
    return m


