import numpy as np
import gzip
import pickle
import cntk as C
from cntk.layers import Dense, Dropout, BatchNormalization
from os.path import expanduser, join
import matplotlib.pyplot as plt
import pandas as pd


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


home = expanduser("~")
dataset_file_path = join(home, "OneDrive/Work/SAFORO Internal Training/Deep learning with CNTK/datasets/creditcard.csv")
__, abnormal_x = load_creditcardfraud(dataset_file_path)

scorer = C.load_model("autoencoder_creditcard.model")

total_image_samples = abnormal_x.shape[0]
test_mb = 1
total_mb = int(total_image_samples / test_mb)

result = []
for i in range(total_mb):
    lbound, ubound = i * test_mb, (i + 1) * test_mb
    data_x = abnormal_x[lbound:ubound]
    result.append(scorer.eval({scorer.arguments[0]: data_x}))

# The one thing i didn't do is that i never hold out
# normal samples to test and see if i got any false positives
threshold = 50
nb_detected = sum(i > threshold for i in result)
accuracy = nb_detected / total_image_samples
print(f"Number of abnormalies are {total_image_samples}")
print(f"{nb_detected} abnormalies detected at {threshold}")
print(f"Accuracy is {accuracy}")