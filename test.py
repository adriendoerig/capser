import numpy as np
import matplotlib.pyplot as plt

label_to_shape = {0: 'vernier', 1: 'squares', 2: 'circles', 3: '7stars', 4: 'stuff'}
shape_to_label = dict([[v, k] for k, v in label_to_shape.items()])

print(len(label_to_shape))

