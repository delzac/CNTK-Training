import csv
import numpy as np
import pandas as pd


def simple_read(data_filepath):
    with open(data_filepath, 'r') as f:
        reader = csv.reader(f)

        next(reader)  # removes header
        data_x = []
        data_y = []
        for line in reader:
            data_x.append(line[:-1])
            data_y.append(line[-1])

    data_x = np.array(data_x)  # converts list into np.array
    data_y = np.array(data_y)

    assert data_y.shape[0] == data_x.shape[0]
    return data_x, data_y


def load_creditcardfraud(data_filepath):
    df = pd.read_csv(data_filepath)
    df = df.drop(['Time'], axis=1)

    df_normal = df[df['Class'] == 0]
    df_abnormal = df[df['Class'] == 1]

    data_df_normal = df_normal.drop(['Class'], axis=1)
    data_df_abnormal = df_abnormal.drop(['Class'], axis=1)

    normal_x = data_df_normal.values
    abnormal_x = data_df_abnormal
    print(f"There are {normal_x.shape[0]} normal and {abnormal_x.shape[0]} abnormal samples")
    return normal_x, abnormal_x
