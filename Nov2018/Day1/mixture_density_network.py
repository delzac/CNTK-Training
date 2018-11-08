from Nov2018.helper import generate_sine_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import cntk as C
import math
from cntk.layers import Dense
import random


def mdn_coeff(output, nmix: int):
    """ Gets the coefficients of mdn. Assumes 1d output """
    alpha = C.softmax(C.slice(output, 0, 0, nmix))
    mu = C.slice(output, 0, nmix, nmix * 2)
    sigma = C.exp(C.slice(output, 0, nmix * 2, nmix * 3))
    # print(alpha.shape, mu.shape, sigma.shape)
    return alpha, mu, sigma


def mdn_1d_phi(mu, sigma, target):
    c = 1
    exp = C.exp(-C.square((target - mu) / sigma) / 2)
    factor = C.pow(2 * C.Constant(math.pi), c / 2) * C.pow(sigma, c)
    factor = C.reciprocal(factor)
    return factor * exp


# @C.Function
def mdn_loss(output, target, nmix: int):
    alpha, mu, sigma = mdn_coeff(output, nmix)
    phi = mdn_1d_phi(mu, sigma, target)
    L = C.reduce_prod(alpha * phi)
    E = -1 * C.log(L)
    return E


def mdn_3gaussian(pred, nb_mix=3):
    mu, sigma, alpha = get_mixture_coeff(pred, nb_mix=nb_mix)
    m = np.sum(alpha * mu, axis=-1)
    s = np.sum(alpha * sigma, axis=-1)
    return m, s


def make_range(m, s):
    return m + s, m, m - s


def generate_ensemble(pred, nb_mix=3):
    mu, sigma, alpha = get_mixture_coeff(pred, nb_mix=nb_mix)

    indices = []
    for rows in alpha.tolist():
        p = random.random()
        for i, a in enumerate(rows):
            p -= a
            if p <= 0:
                indices.append(i)
                break

    m = mu[[i for i in range(mu.shape[0])], indices]
    s = sigma[[i for i in range(sigma.shape[0])], indices]
    s = np.abs(s)
    result = np.random.normal(m, s)
    return result


def get_mixture_coeff(pred, nb_mix=3):
    alpha = np.exp(pred[:, :nb_mix])
    alpha = alpha / np.sum(alpha, axis=-1)[:, None]
    sigma = np.exp(pred[:, int(2 * nb_mix): int(3 * nb_mix)])
    mu = pred[:, nb_mix:int(2 * nb_mix)]
    return mu, sigma, alpha


x, y = generate_sine_data(1000)

x, y = x[:, None].astype(np.float32), y[:, None].astype(np.float32)
x, y = shuffle(x, y)

input_tensor = C.input_variable(1, name="input_tensor")
target_tensor = C.input_variable(1, name="target_tensor")

# model
inner = Dense(20, activation=C.relu)(input_tensor)
inner = Dense(20, activation=C.relu)(inner)
# inner = Dense(20, activation=C.relu)(inner)
# inner = Dense(20, activation=C.relu)(inner)


prediction_tensor = Dense(9, activation=None)(inner)

loss = mdn_loss(prediction_tensor, target_tensor, nmix=3)

# sgd_momentum = C.momentum_sgd(prediction_tensor.parameters, 0.001, 0.9)
adam = C.adam(prediction_tensor.parameters, 0.001, 0.9)  # optimiser

trainer = C.Trainer(prediction_tensor, (loss, ), [adam])

# training loop
num_epoch = 100
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

prediction = np.array(prediction)
m, s = mdn_3gaussian(prediction, nb_mix=3)
p1, p2, p3 = make_range(m, s)
print(prediction.shape, p1.shape, p2.shape, x.shape)
sample = generate_ensemble(prediction, nb_mix=3)
sample1 = generate_ensemble(prediction, nb_mix=3)

plt.scatter(x, y, s=2, c='blue')
plt.scatter(x, p1, s=2, c='red')
plt.scatter(x, p2, s=2, c='red')
plt.scatter(x, p3, s=2, c='red')
plt.scatter(x, sample, s=2, c='green')
plt.scatter(x, sample1, s=2, c='green')

plt.show()