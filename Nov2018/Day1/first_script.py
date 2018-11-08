import cntk as C
import numpy as np


"""
Logic Gate

i1 i2  out
0  0   1
1  0   0
0  1   0
1  1   0

"""


def addition(a, b):
    result = a + b
    print('Inside python function', result)
    return result


a = C.input_variable(1)
b = C.input_variable(1)

c = addition(a, b)

n1 = np.random.random((1,))
n2 = np.random.random((1,))

print((c.eval({a: n1, b: n2})))
print((c.eval({a: n1, b: n2})))
