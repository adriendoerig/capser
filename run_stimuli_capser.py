#### FOR NOW, A FAILED ATTEMPT. this is done in capser_2. ####


import tensorflow as tf
import numpy as np
from data_handling_functions import make_stimuli
from capsule_functions import create_multiple_masked_inputs, decoder_with_mask

model = 'capser_1d'

image_batch, image_labels = make_stimuli(stim_type='square', offset='left')

with tf.Session() as sess:

    # First let's load meta graph and restore weights
    model_files = './'+model+' files/'
    saver = tf.train.import_meta_graph(model_files+'model_'+model+'.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'+model+' files'))
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name('X:0')
    y = graph.get_tensor_by_name('y:0')
    mask_with_labels = graph.get_tensor_by_name('mask_with_labels:0')
    [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
    # caps2_output = graph.get_tensor_by_name('primary_to_first_fc/rba_output:0')
    caps2_output = sess.run([caps2_output],
                          feed_dict={X: image_batch,
                                     y: image_labels,
                                     mask_with_labels: True})
    im_size = (60,128)
    caps2_n_caps = 8
    caps2_n_dims = 10

    print(caps2_output)

    with tf.name_scope('Visualize_colored_capsule_outputs'):
        caps_to_visualize = range(caps2_n_caps)
        decoder_inputs = create_multiple_masked_inputs(caps_to_visualize,caps2_output,caps2_n_caps,caps2_n_dims,mask_with_labels)

        # decoder layer sizes
        n_hidden1 = 512
        n_hidden2 = 1024
        n_output = im_size[0] * im_size[1]

        # run decoder
        decoder_outputs = decoder_with_mask(decoder_inputs, n_hidden1, n_hidden2, n_output)
        decoder_output_images = tf.reshape(decoder_outputs, [-1, im_size[0], im_size[1], caps_to_visualize])

        decoder_outputs_overlay = np.zeros(shape=(im_size[0],im_size[1],3))
        for cap in caps_to_visualize:
            color_mask = [1,0,0]
            decoder_outputs_overlay += np.reshape(decoder_output_images[:, :, :, cap], image_batch.shape[0])*color_mask
        tf.summary.image('decoder_outputs_overlay', decoder_outputs_overlay)

    # summary writer
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('run_stimuli_' + model + '_logdir', sess.graph)

    decoder_output_images, summ = sess.run(
        [decoder_output_images, summary],
        feed_dict={X: image_batch,
                   y: image_labels,
                   mask_with_labels: False})

    writer.add_summary(summ, 1)