import cntk as C
from cntk.layers import Recurrence
import numpy as np


# You do not need a parameter tensor
@C.Function
def cumsum(h, x):
    return h + x


a = C.sequence.input_variable(1)
output_tensor = Recurrence(cumsum)(a)

# sequence of 1 dim with seq_length of 6
n = np.array([0, 1, 2, 3, 4, 5])[None, :, None]
assert n.shape == (1, 6, 1)

results = output_tensor.eval({a: n})
assert np.all(results[0] == np.array([0, 1, 3, 6, 10, 15])[:, None])
print("Success!! :)")
