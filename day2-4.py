import numpy as np
import gzip
import pickle
import cntk as C
from cntk.layers import Dense, Dropout, BatchNormalization
from os.path import expanduser, join


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
x_train, __, y_train, __ = load_data(dataset_file_path)

anomaly_test_case_random = np.random.random((4, 28, 28))
anomaly_test_case_white = np.ones_like(anomaly_test_case_random)
anomaly_test_case_black = np.zeros_like(anomaly_test_case_white)

anomaly_test_case = np.concatenate((anomaly_test_case_random, anomaly_test_case_white, anomaly_test_case_black)).astype(np.float32)
assert anomaly_test_case.ndim == 3
assert anomaly_test_case.shape[1:] == (28, 28)

total_number_of_samples = x_train.shape[0]
mini_batch_size = 16
total_train_batches = int(total_number_of_samples / mini_batch_size)

# ==================================================
# ---------------- MODEL BUIlDING ------------------
# ==================================================
# Only CNTK Computation Nodes to be used from here on

image_tensor = C.input_variable((28, 28), name="image")

# Introducing dropout layer encourages the model not to learn spurious relationships i.e. not overfit
encoded_tensor = Dense(shape=(8, ), activation=C.tanh)(image_tensor)

# activation is None because loss function already has softmax!!
decoded_tensor = Dense(shape=(28, 28), activation=C.sigmoid, name='reconstructed_image')(encoded_tensor)
anomaly_scorer = C.squared_error(decoded_tensor, image_tensor, name="anomaly_score")

loss = C.squared_error(decoded_tensor, image_tensor)  # Used to learn parameters

# Always a good idea to add some regularisation to your optimiser
lr = [0.001] * 5 + [0.0001] * 10 + [0.00005]
lr_schedule = C.learning_parameter_schedule(0.001, minibatch_size=mini_batch_size, epoch_size=100)
adam = C.adam(decoded_tensor.parameters, lr_schedule, 0.912, l2_regularization_weight=0.001)  # optimisation scheme
pp = C.logging.ProgressPrinter(freq=10, log_to_file="autoencoder_log.txt")
trainer = C.Trainer(decoded_tensor, (loss, ), [adam], progress_writers=pp)
# ==================================================
# ==================================================

# epoch is frequently defined as one entire sweep of the training set
nb_epoches = 5
for e in range(nb_epoches):
    for mb_idx in range(total_train_batches):

        lbound, ubound = mb_idx * mini_batch_size, (mb_idx + 1) * mini_batch_size
        data_x = x_train[lbound: ubound, ...]  # first dimension is the batch axis
        data_y = y_train[lbound: ubound, ...]

        trainer.train_minibatch({image_tensor: data_x})

        print(f"loss: {trainer.previous_minibatch_loss_average:.4f}")

    trainer.summarize_training_progress()

# Evaluation of test case
anomaly_scores = anomaly_scorer.eval({image_tensor: anomaly_test_case})
print(f"Anomaly score on handcrafted anomaly test cases are {anomaly_scores}")

anomaly_scorer.save("autoencoder.model")
