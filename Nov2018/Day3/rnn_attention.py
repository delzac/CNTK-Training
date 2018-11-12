import cntk as C
from cntk.layers import Recurrence, LSTM, UnfoldFrom, Dense
from cntk.layers.models import AttentionModel
import numpy as np


def auto_regressive(decoder):

    @C.Function
    def model(thought_vector, dyn_axis):
        unfold = UnfoldFrom(decoder >> C.hardmax,
                            until_predicate=lambda w: w[0],
                            length_increase=2.)
        return unfold(thought_vector, dynamic_axes_like=dyn_axis)

    return model


def decoder_model(attention_dim: int, decoder_dim: int, num_class: int):
    attention = AttentionModel(attention_dim=attention_dim)
    lstm = LSTM(decoder_dim)
    dense = Dense(num_class)

    @C.Function
    def decoder(encoded_tensor, target_tensor):

        @C.Function
        def lstm_with_attention(h, c, x):
            # attention is inside lstm cell as we want it to
            # attend differently every time step (dynamic)
            # 'attended_encoded' is a weighted sum of encoded tensor
            attended_encoded = attention(encoded_tensor, h)
            xx = C.splice(attended_encoded, x)
            return lstm(h, c, xx)

        out = Recurrence(lstm_with_attention)(target_tensor)
        prediction = dense(out)
        return prediction

    return decoder


source_tensor = C.sequence.input_variable(10)

encoder = Recurrence(LSTM(40))
encoded = encoder(source_tensor)
thought_vector = C.sequence.last(encoded)  # this is the seed to the decoder

decoder = decoder_model(30, 20, 10)
decoded = decoder(encoded, source_tensor)
# decoded = auto_regressive(decoder)(thought_vector, source_tensor)  # << same as source tensor
assert decoded.shape == source_tensor.shape
print(encoded.shape, decoded.shape)

n = np.random.random((1, 12, 10))
results = decoded.eval({source_tensor: n})
print(results[0].shape)
