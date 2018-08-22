import os
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

SUBJECT_ID = 0
if not os.path.exists("./human_exp_results"):
   os.mkdir("./human_exp_results")

# use the first "free" subject_ID
while os.path.exists("./human_exp_results/subject_"+str(SUBJECT_ID)+'.pkl'):
    SUBJECT_ID += 1

def press(event):
    print('press', event.key)
    sys.stdout.flush()
    if event.key == 'left' or event.key == 'right':
        responses[stim_ID] = event.key
        plt.close()


random.seed(random.randint(0, 1000))

# folder with the images
# Lpath = './crowding_output/left'
# Rpath = './crowding_output/right'
im_path = './crowding_output'

# list files in folders
# filesL = os.listdir(Lpath)
# filesR = os.listdir(Rpath)
files_im = os.listdir(im_path)

# get all stimulus type
stim_types = [stim[:-9] for stim in files_im]
stim_types = list(set(stim_types))
# print(stim_types,'****')
# print(len(stim_types))

# make a list with all stimuli in random order and create a response collector
# all_stims = filesL + filesR
random.shuffle(files_im)
# print(files_im)

responses = np.zeros_like(files_im)

# main loop
stim_ID = 0
score = np.zeros_like(responses, dtype=int)

for stim in files_im:
    img = plt.imread(im_path + '/' + stim)
    # FIRST, is it a left or right stim?
    if stim[-8] == 'L':
        correct_resp = 'left'
        # img = plt.imread(im_path + '/' + stim)
    elif stim[-8] == 'R':
        correct_resp = 'right'
        # img = plt.imread(im_path + '/' + stim)
    else:
        raise Exception(
            'There is a problem when determining if this offset is L or R. Please check stimulus file names. \n'
            'File names should be of form XXX_L_NN.png')

    # present the image
    fig = plt.figure()
    # if correct_resp is 'left':
    #    img = plt.imread(Lpath + '/' + stim)
    # else:
    #    img = plt.imread(Rpath + '/' + stim)

    plt.title(
        str(stim_ID) + '/' + str(len(files_im)) + '\nDo you think the image was made from a LEFT of a RIGHT stimulus? \n Enter left arrow for LEFT or right arrow for RIGHT.')
    plt.imshow(img)
    fig.canvas.mpl_connect('key_press_event', press)
    manager = plt.get_current_fig_manager()
    #manager.window.setGeometry(650, 350, 600, 370)
    plt.show()

    # ask for reesponse and make sure it is valid
    # responses[stim_ID] = input('Do you think the image was made from a LEFT of a RIGHT stimulus? Enter 1 for LEFT or 3 for RIGHT.')
    # while responses[stim_ID] != '1' and responses[stim_ID] != '3':
    #     responses[stim_ID] = input('You did not enter a valid key, please enter 1 for LEFT or 3 for RIGHT.')

    # check if this is the correct response
    if responses[stim_ID] == correct_resp:
        score[stim_ID] = 1

    stim_ID += 1

final_results = np.zeros(len(stim_types))
stim_occurences = np.zeros_like(final_results)
results_dict = {}
for stim in range(len(stim_types)):
    for im in range(len(files_im)):
        print(files_im[im])
        if stim_types[stim] == files_im[im][:-9]:
            stim_occurences[stim] += 1
            final_results[stim] += score[im]
    results_dict[stim_types[stim]] = final_results[stim] / stim_occurences[stim] * 100

with open("./human_exp_results/subject_" + str(SUBJECT_ID) + '.pkl', 'wb') as f:
    pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)

# final_mean_performance = final_results/stim_occurences
# print(final_mean_performance)

# if not os.path.exists("./human_exp_results"):
#    os.mkdir("./human_exp_results")
# np.save("./human_exp_results/subject_"+str(SUBJECT_ID), final_mean_performance)
