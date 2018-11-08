import cntk as C

tensor = C.input_variable(9)
mu = C.slice(tensor, 0, 0, 3)
alpha = C.softmax(C.slice(tensor, 0, 3, 6))
sigma = C.exp(C.slice(tensor, 0, 6, 9))

print(tensor)
print(mu)
print(alpha)
print(sigma)