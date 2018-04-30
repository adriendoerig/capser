import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from capsule_functions import safe_norm


labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
print(labels)
print(labels.shape)
labels = np.tile(labels,[2,1])
print(labels)
print(labels.shape)
labels = np.transpose(labels)
print(labels)
print(labels.shape)
