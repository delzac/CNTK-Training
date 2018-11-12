import cntk as C
from cntk.layers import RNNStep, Recurrence, Dense
from Nov2018.helper import load_data


data_filepath = "C:/Users/delzac/OneDrive/Work/SAFORO Internal Training" \
                "/Deep learning with CNTK/datasets/mnist/mnist.pkl.gz"

x_train, x_test, y_train, y_test = load_data(data_filepath)


# image_tensor = C.input_variable(shape=(28, 28))  # We are not using this
input_tensor = C.sequence.input_variable(shape=(28, ))
target_tensor = C.input_variable(shape=(10, ))

hidden_layer = Recurrence(RNNStep(shape=32))(input_tensor)
prediction_tensor = Dense(shape=10, activation=None)(C.sequence.last(hidden_layer))

loss = C.cross_entropy_with_softmax(prediction_tensor, target_tensor)
metric = C.classification_error(prediction_tensor, target_tensor)

adam = C.adam(prediction_tensor.parameters, 0.05, 0.9)
trainer = C.Trainer(prediction_tensor, (loss, metric), [adam])

num_epoches = 100
minibatch_size = 8

for epoch in range(num_epoches):
    for i in range(0, len(x_train), minibatch_size):
        lbound, ubound = i, i + minibatch_size
        trainer.train_minibatch({input_tensor: x_train[lbound:ubound, ...],
                                 target_tensor: y_train[lbound:ubound, ...]})

        print(f"loss: {trainer.previous_minibatch_loss_average:.4f}  "
              f"metric: {trainer.previous_minibatch_evaluation_average:.4f}")
