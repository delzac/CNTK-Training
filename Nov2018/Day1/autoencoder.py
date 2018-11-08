import cntk as C
import numpy as np
from cntk.layers import Dense
from Nov2018.helper import load_creditcardfraud


data_filepath = "C:/Users/delzac/OneDrive/Work/SAFORO Internal Training/Deep learning with CNTK/datasets/creditcard.csv"

normal_x, abnormal_x = load_creditcardfraud(data_filepath)
assert normal_x.shape[-1] == 29

input_tensor = C.input_variable(normal_x.shape[-1], name='input_tensor')

# Build model
encoded1 = Dense(15, activation=C.relu)(input_tensor)
latent = Dense(8, activation=C.relu)(encoded1)
decoded1 = Dense(15, activation=C.relu)(latent)

reconstructed = Dense(normal_x.shape[-1], activation=None)(decoded1)

loss = C.squared_error(reconstructed, input_tensor)  # reconstruction error

adam = C.adam(reconstructed.parameters, 0.001, 0.9)

trainer = C.Trainer(reconstructed, (loss, ), [adam])

loss_history = []

num_epoch = 100
minibatch_size = 16
# training loop below
for epoch in range(num_epoch):
    for i in range(0, normal_x.shape[0], minibatch_size):
        lbound, ubound = i, i + minibatch_size
        trainer.train_minibatch({input_tensor: normal_x[lbound:ubound, ...]})

        print(f"loss: {trainer.previous_minibatch_loss_average:.3f}")
        loss_history.append(trainer.previous_minibatch_loss_average)

reconstructed.save("")

new_model = C.load_model("")
