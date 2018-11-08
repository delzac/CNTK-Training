from Nov2018.helper import generate_sine_data
import matplotlib.pyplot as plt
import cntk as C
import numpy as np
from cntk.layers import Dense
from sklearn.utils import shuffle

x, y = generate_sine_data(100)

x, y = x[:, None].astype(np.float32), y[:, None].astype(np.float32)
x, y = shuffle(x, y)

input_tensor = C.input_variable(1, name="input_tensor")
target_tensor = C.input_variable(1, name="target_tensor")

# model
inner = Dense(16, activation=C.relu)(input_tensor)
inner = Dense(16, activation=C.relu)(inner)
inner = Dense(16, activation=C.relu)(inner)
inner = Dense(16, activation=C.relu)(inner)


prediction_tensor = Dense(1, activation=None)(inner)

loss = C.squared_error(prediction_tensor, target_tensor)

# sgd_momentum = C.momentum_sgd(prediction_tensor.parameters, 0.001, 0.9)
adam = C.adam(prediction_tensor.parameters, 0.01, 0.9)  # optimiser

trainer = C.Trainer(prediction_tensor, (loss, ), [adam])

# training loop
num_epoch = 1000
minibatch_size = 10

for epoch in range(num_epoch):
    for i in range(0, x.shape[0], minibatch_size):
        lbound, ubound = i, i + minibatch_size
        x_mini = x[lbound:ubound]
        y_mini = y[lbound:ubound]
        trainer.train_minibatch({input_tensor: x_mini,
                                 target_tensor: y_mini})

        print(f"loss: {trainer.previous_minibatch_loss_average}")

prediction = prediction_tensor.eval({input_tensor: x})

plt.scatter(x, y)
plt.scatter(x, prediction)
plt.show()
