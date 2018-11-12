import cntk as C
from cntk.layers import Recurrence, RNNStep


def RNN(shape: int):

    wx = C.Parameter((C.InferredDimension, shape), init=C.glorot_uniform())
    wh = C.Parameter((C.InferredDimension, shape), init=C.glorot_uniform())

    @C.Function
    def step_function(previous_state1, current_data):
        xx = C.times(current_data, wx)
        hx = C.times(previous_state1, wh)
        return C.tanh(xx + hx)  # becomes previous_state1 in next timestep

    return step_function


input_tensor = C.sequence.input_variable(10)
hidden = Recurrence(RNN)(input_tensor)

print(hidden.shape)

