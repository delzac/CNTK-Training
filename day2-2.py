import numpy as np
import gzip
import pickle
import cntk as C
from cntk.layers import Dense
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
x_train, x_test, y_train, y_test = load_data(dataset_file_path)

total_number_of_samples = x_train.shape[0]
mini_batch_size = 4
total_train_batches = int(total_number_of_samples / mini_batch_size)

total_nb_test_samples = int(x_test.shape[0] / 10)  # divided by 10 to reduce test dataset
total_test_batches = int(total_nb_test_samples / mini_batch_size)
# ==================================================
# ---------------- MODEL BUIlDING ------------------
# ==================================================
# Only CNTK Computation Nodes to be used from here on

image_tensor = C.input_variable((28, 28), name="image")
ground_truth_tensor = C.input_variable(10, name="ground_truth")

hidden_layer_output = Dense(shape=(16, ), activation=C.tanh)(image_tensor)

# activation is None because loss function already has softmax!!
output_tensor = Dense(shape=(10, ), activation=None)(hidden_layer_output)

loss = C.cross_entropy_with_softmax(output_tensor, ground_truth_tensor)  # Used to learn parameters
metric = C.classification_error(output_tensor, ground_truth_tensor)  # Not used to learn parameters

adam = C.adam(output_tensor.parameters, 0.001, 0.912, l2_regularization_weight=0.001)  # optimisation scheme
pp = C.logging.ProgressPrinter(freq=0, log_to_file="mnist_dense_log.txt")
trainer = C.Trainer(output_tensor, (loss, metric), [adam], progress_writers=pp)
# ==================================================
# ==================================================

# epoch is frequently defined as one entire sweep of the training set
nb_epoches = 5
for e in range(nb_epoches):
    for mb_idx in range(total_train_batches):

        lbound, ubound = mb_idx * mini_batch_size, (mb_idx + 1) * mini_batch_size
        data_x = x_train[lbound: ubound, ...]  # first dimension is the batch axis
        data_y = y_train[lbound: ubound, ...]

        trainer.train_minibatch({image_tensor: data_x,
                                  ground_truth_tensor: data_y})

        # print(f"loss: {trainer.previous_minibatch_loss_average:.4f}, metric: {trainer.previous_minibatch_evaluation_average}")

    trainer.summarize_training_progress()

    for test_idx in range(total_test_batches):
        lbound, ubound = test_idx * mini_batch_size, (test_idx + 1) * mini_batch_size

        trainer.test_minibatch({image_tensor: x_test[lbound:ubound, ...],
                                ground_truth_tensor: y_test[lbound:ubound, ...]})

    trainer.summarize_test_progress()
output_tensor.save("mnist_dense.model")

