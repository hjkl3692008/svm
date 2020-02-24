import pandas as pd
import os
import numpy as np

import svm_tools as st


# get cwd
def get_cwd():
    return os.getcwd()


# join path
def join_path(*args):
    path = ''
    for v in args:
        path = os.path.join(path, v)
    return path


basic_path = join_path(get_cwd(), os.path.pardir, 'data')


# load
def load_csv(path):
    data = pd.read_csv(path)
    return data


# save csv
def save_csv(data, path):
    df = pd.DataFrame(data)
    df.to_csv(path, index=False, sep=',')


# judge whether file exist
def is_exist(t, name):
    full_path = join_path(basic_path, t, name)
    flag = os.path.exists(full_path)
    return flag


# load breast-cancer-wisconsin data
def load_cancer(trans=True, is_drop=True):
    path = join_path(basic_path, 'breast-cancer-wisconsin.data')
    names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
              'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
              'Mitoses', 'Class']
    data = pd.read_csv(path, names=names)
    if is_drop:
        data = drop_question(data)
    if trans:
        data = np.array(data)
    return data


# drop drop_question
def drop_question(df):
    df = df[~df.isin(['?'])]
    df = df.dropna()
    df['Bare Nuclei'] = pd.to_numeric(df['Bare Nuclei'])
    return df

