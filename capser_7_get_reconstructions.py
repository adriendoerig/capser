from capser_7_model_fn_temp import *
from capser_7_input_fn import *
import logging
import numpy as np
import matplotlib.pyplot as plt

save_np_arrays = True  # true -> np array, false -> png images

# reproducibility
tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

# directory management
print('#################### MODEL_NAME_VERSION: ' + MODEL_NAME + ' ####################')

# create estimator for model (the model is described in capser_7_model_fn)
capser = tf.estimator.Estimator(model_fn=model_fn_temp, params={'model_batch_size': batch_size}, model_dir=checkpoint_path)

logging.getLogger().setLevel(logging.CRITICAL)  # to show less info in the console

common_dir = image_output_dir+'reconstructions_'+str(n_steps)+'_noise_'+str(test_noise_level)+'_shape_size_'+str(shape_size)
if not os.path.exists(common_dir):
    os.mkdir(common_dir)

n_expt_batches = 1
for category in test_stimuli.keys():

    print('CREATING RECONSCTRUCTIONS FOR ' + category)
    if save_np_arrays:
        this_dir = common_dir + '/np_arrays_' + category
    else:
        this_dir = common_dir+'/'+category
    if not os.path.exists(this_dir):
        os.mkdir(this_dir)

    for stim in range(3):

        for batch in range(n_expt_batches):

            capser_out = list(capser.predict(input_fn=lambda: input_fn_config(data_path+'/'+category+'/test_'+category+str(stim)+'.tfrecords')))
            reconstructions = np.array([p["reconstructions"] for p in capser_out])
            vernier_offsets = np.array([p["vernier_offsets"] for p in capser_out])

            for im in range(reconstructions.shape[0]):

                if im < 10:
                    this_ID = '000'+str(im)
                elif im <100:
                    this_ID = '00' + str(im)
                elif im < 1000:
                    this_ID = '0' + str(im)
                else:
                    this_ID = str(im)

                # normalize image to fit [0,255] (otherwise colors get all fucked up)
                this_reconstruction = reconstructions[im, :, :, :]/np.amax(reconstructions[im,:,:,:])
                if vernier_offsets[im] == 0:
                    this_offset = 'L'
                else:
                    this_offset = 'R'

                if save_np_arrays:
                    curr_dir = this_dir+'/'+str(stim)
                    if not os.path.exists(curr_dir):
                        os.mkdir(curr_dir)
                    np.save(curr_dir+'/'+category+str(stim)+'_'+this_offset+'_'+this_ID, this_reconstruction)  # additional arguments make sure that we have only the picture, no frame, axes, etc.

                else:
                    plt.figure()
                    plt.imshow(this_reconstruction)
                    plt.axis('off')
                    plt.savefig(this_dir+'/'+category+str(stim)+'_'+this_offset+'_'+this_ID+'.png', transparent=True, bbox_inches='tight', pad_inches=0)  # additional arguments make sure that we have only the picture, no frame, axes, etc.
                    plt.close()