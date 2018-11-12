import cntk as C
from cntk.layers import Recurrence
import numpy as np
import warnings
warnings.filterwarnings()

# You do not need a parameter tensor
@C.Function
def seq_length(h, x):
    return 1 + h + x * 0


a = C.sequence.input_variable(1)
output_tensor = C.sequence.last(Recurrence(seq_length)(a))

# sequence of 1 dim with seq_length of 6
n = np.array([0, 1, 2, 3, 4, 5])[None, :, None]
assert n.shape == (1, 6, 1)

results = output_tensor.eval({a: n})
assert results[0][0] == 6
print("Success!! :)")
