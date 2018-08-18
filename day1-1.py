import cntk as C
import numpy as np


model = C.load_model("LogicGate.model")
datax = np.array([[0, 1]]).astype(np.float32)
assert datax.shape == (1, 2)

output = model.eval({model.arguments[0]: datax})
print(output)

"""
Some people in the deep learning world calls AI as
a new way of writing software. Indeed, this particular example
illustrate this very clearly.

Instead of

if input1 == 0 and input2 == 0:
    return 1
elif input1 == 1 and input2 ==0:
    return 0

we simply use machine learning to do the above, all without writing a
line of code!
"""
