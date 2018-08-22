import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


# plot bar width & colors
width = .3333  # the width of the bars
color0 = (179. / 255, 179. / 255, 179. / 255)
color1 = (125. / 255, 163. / 255, 170. / 255)
color2 = (209. / 255, 184. / 255, 148. / 255)
color3 = (87. / 255, 140. / 255, 169. / 255)
color4 = (203. / 255, 123. / 255, 123. / 255)
color5 = (141. / 255, 179. / 255, 203. / 255)
color6 = (229. / 255, 196. / 255, 148. / 255)
color7 = (100. / 255, 170. / 255, 180. / 255)


N_SUBJECTS = 0
while os.path.exists("./human_exp_results/subject_"+str(N_SUBJECTS)+'.pkl'):
    N_SUBJECTS += 1


created_mean_dict = 0
for SUBJECT_ID in range(N_SUBJECTS):
    with open("./human_exp_results/subject_" + str(SUBJECT_ID) + '.pkl', 'rb') as f:
        results_dict = pickle.load(f)
        these_keys = []
        these_values = []
        for key, value in sorted(results_dict.items()):  # alphabetical order
            these_keys.append(key)
            these_values.append(value)

    if not created_mean_dict:
        mean_dict = results_dict
        for key, value in mean_dict.items():
            mean_dict[key] = mean_dict[key] / N_SUBJECTS
        created_mean_dict = 1
    else:
        for key, value in mean_dict.items():
            mean_dict[key] += results_dict[key] / N_SUBJECTS

    ####### PLOT RESULTS #######

    N = len(results_dict)
    ind = np.arange(N)  # the x locations for the groups
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, these_values, width, color=[color0, color1, color1, color2, color2, color3, color3, color4, color4, color5, color5, color6, color6, color7, color7])
    # add some text for labels, title and axes ticks, and save figure
    ax.set_ylabel('Percent correct')
    ax.set_title('Human performance detecting vernier offset - SUBJECT ' + str(SUBJECT_ID))
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(these_keys)
    ax.plot([-0.3, N], [50, 50], 'k--')  # chance level cashed line
    plt.savefig("./human_exp_results/subject_" + str(SUBJECT_ID) + '.png')
    # plt.show()
    plt.close()

created_std_dict = 0

for SUBJECT_ID in range(N_SUBJECTS):
    with open("./human_exp_results/subject_" + str(SUBJECT_ID) + '.pkl', 'rb') as f:
        results_dict = pickle.load(f)

    if not created_std_dict:
        std_dict = results_dict
        for key, value in std_dict.items():
            std_dict[key] = (results_dict[key] - mean_dict[key]) ** 2 / N_SUBJECTS
        created_std_dict = 1
    else:
        for key, value in std_dict.items():
            std_dict[key] += (results_dict[key] - mean_dict[key]) ** 2 / N_SUBJECTS

for key, value in std_dict.items():
    std_dict[key] = np.sqrt(std_dict[key])

# for alphabetical order in plot
these_keys_mean = []
these_keys_std = []
these_values_mean = []
these_values_std = []
for key, value in sorted(mean_dict.items()):
    these_keys_mean.append(key)
    these_values_mean.append(value)
for key, value in sorted(std_dict.items()):
    these_keys_std.append(key)
    these_values_std.append(value)

N = len(these_values_mean)
ind = np.arange(N)  # the x locations for the groups
fig, ax = plt.subplots()

decodMeans = np.array([0.921875, 0.703125, 0.9375, 0.65625, 0.953125, 0.6875, 0.890625, 0.671875, 0.859375, 0.8125, 0.921875, 0.78125, 0.9375, 0.640625, 0.8125])*100
rects1 = ax.bar(ind-width/2, decodMeans, width, color=[color0, color1, color1, color2, color2, color3, color3, color4, color4, color5, color5, color6, color6, color7, color7], edgecolor='black', hatch='...')
rects2 = ax.bar(ind+width/2, these_values_mean, width, color=[color0, color1, color1, color2, color2, color3, color3, color4, color4, color5, color5, color6, color6, color7, color7], yerr=these_values_std, edgecolor='black')


# add some text for labels, title and axes ticks, and save figure

#decod
#
#
ax.set_ylabel('Percent correct')
ax.set_title("Machine & human decoders' performance detecting vernier offset - MEAN")
ax.set_xticks(ind)
plt.legend(['Machine decoder', 'Human decoders'])
ax.set_xticklabels(these_keys_mean)

chance_line, = ax.plot([-2*width, N], [50, 50], '#8d8f8c')  # chance level dashed line
chance_line.set_dashes([1, 3, 1, 3])
plt.savefig("./human_exp_results/subject_MEAN.png")
plt.show()
plt.close()

print(these_keys)