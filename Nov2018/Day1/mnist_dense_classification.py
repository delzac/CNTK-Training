import cntk as C
import numpy as np
from cntk.layers import Dense
from Nov2018.helper import load_data

data_filepath = "C:/Users/delzac/OneDrive/Work/SAFORO Internal Training" \
                "/Deep learning with CNTK/datasets/mnist/mnist.pkl.gz"

x_train, x_test, y_train, y_test = load_data(data_filepath)

image_tensor = C.input_variable(shape=(28, 28), name='image_tensor')
target_tensor = C.input_variable(shape=(10, ), name='target_tensor')

hidden1 = Dense(shape=16, activation=C.tanh)(image_tensor)
hidden2 = Dense(shape=10, activation=C.tanh)(hidden1)
output_tensor = Dense(shape=10)(hidden2)  # aka. prediction_tensor

loss = C.cross_entropy_with_softmax(output_tensor, target_tensor)

adam = C.adam(output_tensor.parameters, 0.001, 0.9)

trainer = C.Trainer(output_tensor, (loss, ), [adam])

num_epoch = 1000
minibatch_size = 16
# start a training loop
for epoch in range(num_epoch):
    for i in range(0, x_train.shape[0], minibatch_size):
        lbound, ubound = i, i + minibatch_size
        x_mini = x_train[lbound:ubound, ...]
        y_mini = y_train[lbound:ubound, ...]
        trainer.train_minibatch({image_tensor: x_mini,
                                 target_tensor: y_mini})

        print(f"loss: {trainer.previous_minibatch_loss_average}")

output_tensor.save("yourfilepath")
