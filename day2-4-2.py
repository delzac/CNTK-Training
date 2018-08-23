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
normal_x, abnormal_x = load_creditcardfraud(dataset_file_path)

total_number_of_samples = normal_x.shape[0]
mini_batch_size = 8
total_train_batches = int(total_number_of_samples / mini_batch_size)

# ==================================================
# ---------------- MODEL BUIlDING ------------------
# ==================================================
# Only CNTK Computation Nodes to be used from here on

input_tensor = C.input_variable(29, name="input_tensor")

# Introducing dropout layer encourages the model not to learn spurious relationships i.e. not overfit
encode1 = Dense(shape=(20, ), activation=C.relu)(input_tensor)
encode2 = Dense(shape=(15, ), activation=C.relu)(encode1)
encode3 = Dense(shape=(8, ), activation=C.relu)(encode2)

# activation is None because loss function already has softmax!!
decode1 = Dense(shape=(15,), activation=C.relu)(encode3)
decode2 = Dense(shape=(20,), activation=C.relu)(decode1)
decoded_tensor = Dense(shape=(29,), activation=None, name='reconstructed')(decode2)
anomaly_scorer = C.squared_error(decoded_tensor, input_tensor, name="anomaly_score")

loss = C.squared_error(decoded_tensor, input_tensor)  # Used to learn parameters

# Always a good idea to add some regularisation to your optimiser
lr = [0.001] * 5 + [0.0001] * 10 + [0.00005]
lr_schedule = C.learning_parameter_schedule(0.001, minibatch_size=mini_batch_size, epoch_size=100)
adam = C.adam(decoded_tensor.parameters, lr_schedule, 0.912, l2_regularization_weight=0.001)  # optimisation scheme
pp = C.logging.ProgressPrinter(freq=10, log_to_file="autoencoder_creditcard_log.txt")
trainer = C.Trainer(decoded_tensor, (loss, ), [adam], progress_writers=pp)
# ==================================================
# ==================================================

# epoch is frequently defined as one entire sweep of the training set
nb_epoches = 5
for e in range(nb_epoches):
    for mb_idx in range(total_train_batches):

        lbound, ubound = mb_idx * mini_batch_size, (mb_idx + 1) * mini_batch_size
        data_x = normal_x[lbound: ubound, ...]  # first dimension is the batch axis

        trainer.train_minibatch({input_tensor: data_x})

        # print(f"loss: {trainer.previous_minibatch_loss_average:.4f}")

    trainer.summarize_training_progress()

anomaly_scorer.save("autoencoder_creditcard.model")

# Evaluation of test case
anomaly_scores = anomaly_scorer.eval({input_tensor: abnormal_x[:5]})
print(f"Anomaly score on handcrafted anomaly test cases are {anomaly_scores}")