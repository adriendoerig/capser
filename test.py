from data_handling_functions import make_sets
import matplotlib.pyplot as plt
import numpy as np
a = np.ones((2,3))
good_size = (5,5)
b = np.zeros(good_size, dtype='float32')
b[:a.shape[0], :a.shape[1]] = a
print(a)
print(b)
print(np.less(b.shape,a.shape))
# print(b)

