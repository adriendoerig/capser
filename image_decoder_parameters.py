
image_decoder_data_path = './image_decoder'
source_model_path = './data/BS_64_C1DIM_16_C2DIM_16_LR_0.0005_CONVBN_False_DECODERBN_True_DECONVDECODER_False'
source_image_path = source_model_path + '/output_images/reconstructions_75000_noise_0.1_shape_size_15'

batch_size = 64
buffer_size = 1024*1024
im_size = (45, 100)

n_train_samples = 9984
n_test_samples = 150
