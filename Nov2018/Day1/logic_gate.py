import numpy as np
import cntk as C
from cntk.layers import Dense
from cntk.logging.progress_print import TensorBoardProgressWriter

"""
Logic Gate

i1 i2  out
0  0   1
1  0   0
0  1   0
1  1   0

"""

data_x = np.array([[0, 0],
                   [1, 0],
                   [0, 1],
                   [1, 1]]).astype(np.float32)  # Inputs of the logic gate

data_y = np.array([[1],
                   [0],
                   [0],
                   [0]]).astype(np.float32)  # Groundtruths to the logic gate

assert data_x.shape == (4, 2), data_x.shape
assert data_y.shape == (4, 1), data_y.shape

# Multi-Layer Perceptron: y = w * x + b
x = C.input_variable(2, name="x")  # <== data_x
y = C.input_variable(1, name="y")  # <== data_y

# In any optimisation problem, we have parameters that can
# take on many values. We define these in CNTK as C.parameter().
# Variables that are defined as parameters are what we
# called "trainable weights". i.e. we vary the trainable weights
# to minimise the loss function.

hidden_dim = 5
z = Dense(shape=hidden_dim, activation=C.tanh, name="Hidden layer")(x)
z = Dense(shape=1, activation=C.sigmoid, name="output layer")(z)

# Loss/objective/energy function
# loss = C.squared_error(z, y)  # For regression problem
loss = C.binary_cross_entropy(z, y)  # For classification problem
metric = C.Constant(1) - C.equal(C.round(z), y)  # What is this code doing????

# It can be very cumbersome to manually record and keep track
# of the progress of your training. In CNTK, there's something that
# helps you do just that.
pp = C.logging.ProgressPrinter(freq=5, log_to_file="LogicGateLogs.txt")

# Next we need an optimisation scheme to vary the trainable weights
# in a way that reduces the loss function. In any deep learning framework,
# there are many schemes to choose from.
sgd = C.sgd(z.parameters, lr=0.1)  # Stochastic gradient descent
adam = C.adam(z.parameters, lr=0.1, momentum=0.9)  # Adaptive momentum gradient descent

# Trainer class is an abstraction that makes training easy
trainer = C.Trainer(z, (loss, metric), [sgd], progress_writers=pp)

number_of_iterations = 10000
for i in range(number_of_iterations):
    trainer.train_minibatch({x: data_x,
                             y: data_y})

    print(f"{i:04d} loss: {trainer.previous_minibatch_loss_average:.4f} "
          f" metric: {trainer.previous_minibatch_evaluation_average}")

trainer.summarize_training_progress()

print(z.eval({x: data_x}))
z.save("LogicGate.model")
model = C.load_model("LogicGate.model")
