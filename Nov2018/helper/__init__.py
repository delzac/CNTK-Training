import gzip
import pickle
import numpy as np
import random
import pandas as pd


def load_data(filepath: str):
    f = gzip.open(filepath, 'rb')
    (x_train, y_train), (x_test, y_test), __ = pickle.load(f, encoding='latin-1')
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train.reshape((len(x_train), 28, 28))
    x_test = x_test.reshape((len(x_test), 28, 28))

    y_train = ohe_labels(y_train, 10)
    y_test = ohe_labels(y_test, 10)
    print(f"Shape of train data {x_train.shape} and truth {y_train.shape}")
    print(f"Shape of train data {x_test.shape} and truth {y_test.shape}")
    return x_train, x_test, y_train, y_test


def ohe_labels(a, nb_class):
    tmp = np.zeros((a.shape[0], nb_class))
    tmp[np.arange(a.shape[0]), a] = 1
    return tmp.astype(np.float32)


def generate_sine_data(nb_samples: int):
    """ Generates a sine wave dataset """
    e = 2 * np.random.random(size=(nb_samples,)) / 10 - 0.1
    x = np.arange(nb_samples) / nb_samples
    y = x + 0.3 * np.sin(2 * np.pi * x) + e
    print("shape of x and y: {0} {1}".format(x.shape, y.shape))
    return x, y


def generate_variable_10(nb_samples=1000, dim=3):
    """
    Generate a dataset of sequences where sequence length is between 5 to 20.
    Target will be the sum of the sequence.
    """
    r = [random.randint(5, 20) for __ in range(nb_samples)]
    x = [np.random.randint(2, size=(t, dim)) for t in r]
    y = [np.array([arr.sum()]) for arr in x]
    return x, y


def load_creditcardfraud(data_filepath):
    df = pd.read_csv(data_filepath)
    df = df.drop(['Time'], axis=1)

    df_normal = df[df['Class'] == 0]
    df_abnormal = df[df['Class'] == 1]

    data_df_normal = df_normal.drop(['Class'], axis=1)
    data_df_abnormal = df_abnormal.drop(['Class'], axis=1)

    normal_x = np.ascontiguousarray(data_df_normal.values, dtype=np.float32)
    abnormal_x = np.ascontiguousarray(data_df_abnormal.values, dtype=np.float32)
    print(f"There are {normal_x.shape[0]} normal and {abnormal_x.shape[0]} abnormal samples")
    return normal_x, abnormal_x
