import numpy as np
import matplotlib.pyplot as plt
from data_handling_functions import make_stimuli

data, labels, vernier_labels = make_stimuli(stim_type='vernier', folder='crowding_images/vernier_decoder_train_set',
                                            n_repeats=1, resize_factor=0.4, print_shapes=True)
im_size=(60,128)

for im in range(10):
    plt.figure()
    plt.imshow(np.reshape(data[im, :, :, :], im_size))
    plt.title('LABEL = ' + str(vernier_labels[im]))
    plt.show()

