import cntk as C
from cntk.layers import Recurrence, LSTM, UnfoldFrom, Dense
import numpy as np


def auto_regressive(decoder):

    @C.Function
    def model(thought_vector, dyn_axis):
        unfold = UnfoldFrom(decoder >> C.hardmax,
                            until_predicate=lambda w: w[0],
                            length_increase=2.)
        return unfold(thought_vector, dynamic_axes_like=dyn_axis)

    return model


source_tensor = C.sequence.input_variable(10)

encoder = Recurrence(LSTM(40))
encoded = encoder(source_tensor)
thought_vector = C.sequence.last(encoded)  # this is the seed to the decoder

decoder = Recurrence(LSTM(40))
decoded = auto_regressive(decoder)(thought_vector, source_tensor)  # << same as source tensor
reconstructed = Dense(10)(decoded)
assert reconstructed.shape == source_tensor.shape
print(encoded.shape, reconstructed.shape)

n = np.random.random((1, 12, 10))
results = reconstructed.eval({source_tensor: n})
print(results[0].shape)
