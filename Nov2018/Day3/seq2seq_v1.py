import cntk as C
from cntk.layers import Recurrence, LSTM, UnfoldFrom, RecurrenceFrom
import numpy as np
from Nov2018.helper import generate_seq2seq_toy


def greedy_decoder(decoder, sentence_end_index):


    @C.Function
    def model(seed, dyn_axis, h, c):
        unfold = UnfoldFrom(lambda x: decoder(h, c, x) >> C.hardmax,
                            until_predicate=lambda x: x[sentence_end_index],
                            length_increase=2)
        return unfold(seed, dyn_axis)

    return model


vocab_size = 10
sentence_end_index = 0
sentence_start = C.Constant(0, shape=vocab_size)
source_seqs, target_seqs = generate_seq2seq_toy(n=1,
                                                vocab_size=vocab_size,
                                                min_seq_length=5,
                                                max_seq_length=10,
                                                mode='skipcopy')


source_tensor = C.sequence.input_variable(vocab_size)
target_tensor = C.sequence.input_variable(vocab_size)

encoded = Recurrence(LSTM(vocab_size), return_full_state=True)(source_tensor)


decoder = RecurrenceFrom(LSTM(vocab_size))
decoded = decoder(encoded.outputs[0], encoded.outputs[1], target_tensor)
# greedy_decoded = greedy_decoder(decoder, sentence_end_index)(sentence_start, target_tensor, encoded.outputs[0], encoded.outputs[1])

# loss = C.cross_entropy_with_softmax(decoded, target_seqs)

n = np.random.random((1, 10, 10))
print(decoded.eval({source_tensor: n,
                    target_tensor: n}))
