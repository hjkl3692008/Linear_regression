import pandas as pd
import os
import numpy as np


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


# load house data
def load_house(trans=True, is_drop=True):
    p_data = load_csv(join_path(basic_path, 'kc_house_data.csv'))
    if is_drop:
        p_data.drop(['id', 'date', 'zipcode', 'lat'], axis=1)
    if trans:
        p_data = np.array(p_data)
    return p_data


# judge whether file exist
def is_exist(t, name):
    full_path = join_path(basic_path, t, name)
    flag = os.path.exists(full_path)
    return flag


# dataFrame to array
def df2np(data):
    return data.values
