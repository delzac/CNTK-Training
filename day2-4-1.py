import numpy as np
import gzip
import pickle
import cntk as C
from cntk.layers import Dense, Dropout, BatchNormalization
from os.path import expanduser, join
import matplotlib.pyplot as plt


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


home = expanduser("~")
dataset_file_path = join(home, "OneDrive/Work/SAFORO Internal Training/Deep learning with CNTK/datasets/mnist/mnist.pkl.gz")
x_train, x_test, __, __ = load_data(dataset_file_path)

scorer = C.load_model("autoencoder.model")

total_image_samples = x_test.shape[0]
test_mb = 1
total_mb = int(total_image_samples / test_mb)

for i in range(total_mb):
    lbound, ubound = i * test_mb, (i + 1) * test_mb
    data_x = x_test[lbound:ubound]
    result = scorer.eval({scorer.arguments[0]: data_x})

    if result[0] > 120:
        plt.imshow(data_x.reshape((28, 28)))
        plt.show()
