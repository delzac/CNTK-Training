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

input_tensor = C.input_variable(2)
ground_truth_tensor = C.input_variable(1)

hidden_layer_output = Dense(5, activation=C.tanh)(input_tensor)
output_tensor = Dense(1, activation=C.sigmoid)(hidden_layer_output)

loss = C.binary_cross_entropy(output_tensor, ground_truth_tensor)
metric = C.Constant(1) - C.equal(C.round(output_tensor), ground_truth_tensor)
# C.classification_error  <== Cannot use for single column output

pp = C.logging.ProgressPrinter(freq=1, log_to_file="logs.text", gen_heartbeat=True)

adam = C.adam(output_tensor.parameters, 0.1, 0.9)

trainer = C.Trainer(output_tensor, (loss, metric), [adam], progress_writers=pp)

num_epoch = 10000
for e in range(num_epoch):
    trainer.train_minibatch({input_tensor: data_x,
                             ground_truth_tensor: data_y})

trainer.summarize_training_progress()
output_tensor.save("LogicGate.model")
