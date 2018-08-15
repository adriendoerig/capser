import os
import random
import numpy as np
import matplotlib.pyplot as plt
import sys


def press(event):
    print('press', event.key)
    sys.stdout.flush()
    if event.key == '1' or event.key == '3':
        responses[stim_ID] = event.key
        plt.close()

random.seed(random.randint(0, 1000))

# folder with the images
Lpath = './crowding_output/left'
Rpath = './crowding_output/right'

# list files in folders
filesL = os.listdir(Lpath)
filesR = os.listdir(Rpath)

# get all stimulus type
stim_types = [stim[:-7] for stim in filesL]
stim_types = list(set(stim_types))

# make a list with all stimuli in random order and create a response collector
all_stims = filesL + filesR
random.shuffle(all_stims)

responses = np.zeros_like(all_stims)

# main loop
stim_ID = 0
score = np.zeros_like(responses)
for stim in all_stims:

    # first, is it a left or right stim?
    if stim[-8] == 'L':
        correct_resp = '1'
    elif stim[-8] == 'R':
        correct_resp = '3'
    else:
        raise Exception('There is a problem when determining if this offset is L or R. Please check stimulus file names. \n'
                        'File names should be of form XXX_L_NN.png')

    # present the image
    fig = plt.figure()
    if correct_resp is '1':
        img = plt.imread(Lpath + '/' + stim)
    else:
        img = plt.imread(Rpath + '/' + stim)

    plt.title('Do you think the image was made from a LEFT of a RIGHT stimulus? \n Enter 1 for LEFT or 3 for RIGHT.')
    plt.imshow(img)
    fig.canvas.mpl_connect('key_press_event', press)
    plt.show()

    # ask for reesponse and make sure it is valid
    # responses[stim_ID] = input('Do you think the image was made from a LEFT of a RIGHT stimulus? Enter 1 for LEFT or 3 for RIGHT.')
    # while responses[stim_ID] != '1' and responses[stim_ID] != '3':
    #     responses[stim_ID] = input('You did not enter a valid key, please enter 1 for LEFT or 3 for RIGHT.')

    # check if this is the correct response
    if responses[stim_ID] == correct_resp:
        score[stim_ID] = 1

    stim_ID += 1