import numpy as np
from scipy import ndimage, misc
import os
import random


def make_shape_sets(folder='./crowding_images/shapes', image_size=(60, 128), resize_factor=1.0, n_repeats=10,
                    n_valid_samples=100, n_test_samples=144, print_shapes=False):

    min_num_images = 50
    num_images = 0

    image_files = os.listdir(folder)

    train_set = np.ndarray(shape=(n_repeats*len(image_files), image_size[0], image_size[1]),
                           dtype=np.float32)
    train_labels = np.zeros(n_repeats*len(image_files), dtype=np.float32)

    print('loading images from '+folder)

    for image in image_files:
        image_file = os.path.join(folder, image)
        # print('loading '+image_file)

        for rep in range(n_repeats):
            try:
                image_data = ndimage.imread(image_file, mode='L').astype(float)
                image_data = misc.imresize(image_data, resize_factor)

                # crop out a random patch if the image is larger than image_size
                if image_data.shape[0] > image_size[0]:
                    first_row = int((image_data.shape[0] - image_size[0]) / 2)
                    image_data = image_data[first_row:first_row + image_size[0], :]
                if image_data.shape[1] > image_size[1]:
                    first_col = int((image_data.shape[1] - image_size[1]) / 2)
                    image_data = image_data[:, first_col:first_col + image_size[1]]

                # pad to the right size if image is small than image_size (image will be in a random place)
                if any(np.less(image_data.shape, image_size)):
                    pos_x = random.randint(0, max(0, image_size[1] - image_data.shape[1]))
                    pos_y = random.randint(0, max(0, image_size[0] - image_data.shape[0]))
                    padded = np.zeros(image_size, dtype=np.float32)
                    padded[pos_y:pos_y+image_data.shape[0], pos_x:pos_x+image_data.shape[1]] = image_data
                    image_data = padded

                # normalize etc.
                # zero mean, 1 stdev
                image_data = (image_data - np.mean(image_data)) / np.std(image_data)

                # add to training set
                train_set[num_images, :, :] = image_data

                if 'square' in image_file:
                    train_labels[num_images] = 0
                elif 'circle' in image_file:
                    train_labels[num_images] = 1
                elif 'hexagon' in image_file:
                    train_labels[num_images] = 2
                elif 'octagon' in image_file:
                    train_labels[num_images] = 3
                elif 'star' in image_file:
                    train_labels[num_images] = 4
                elif 'line' in image_file:
                    train_labels[num_images] = 5
                elif 'vernier' in image_file:
                    train_labels[num_images] = 6
                    # print('CAREFUL WITH LABELS WHEN DOING DIFFERENT TYPES OF TASK!')
                else:
                    raise Exception(image_file+' is a stimulus of unknown class')

                num_images = num_images + 1

            except (ValueError, IOError, IndexError, OSError):
                print('Could not read:', image_file, '- it\'s ok, skipping.')

    # check enough images could be processed
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    # remove empty entries
    train_set = train_set[0:num_images, :, :]
    # add a singleton 4th dimentsion (needed for conv layers
    train_set = np.expand_dims(train_set, axis=3)

    perm = np.random.permutation(num_images)
    train_set = train_set[perm, :, :, :]
    train_labels = train_labels[perm]
    valid_set = train_set[:n_valid_samples, :, :, :]
    valid_labels = train_labels[:n_valid_samples]
    test_set = train_set[n_valid_samples:n_valid_samples+n_test_samples, :, :, :]
    test_labels = train_labels[n_valid_samples:n_valid_samples+n_test_samples]
    train_set = train_set[n_valid_samples+n_test_samples:, :, :]
    train_labels = train_labels[n_valid_samples+n_test_samples:]

    if print_shapes:
        print('Train set tensor:', train_set.shape)
        print('Valid set tensor:', valid_set.shape)
        print('Test set tensor:', test_set.shape)
        print('Mean:', np.mean(train_set))
        print('Standard deviation:', np.std(train_set))
    return train_set, train_labels, valid_set, valid_labels, test_set, test_labels


def make_stimuli(stim_type='squares', folder='./crowding_images/shapes_simple_test', image_size=(60, 128),
                 resize_factor=1.0, n_repeats=1, print_shapes=False):

    min_num_images = 1
    num_images = 0

    folder = folder+'/'
    image_files = os.listdir(folder)

    image_batch = np.ndarray(shape=(n_repeats*len(image_files), image_size[0], image_size[1]),
                             dtype=np.float32)
    image_labels = np.zeros(n_repeats*len(image_files), dtype=np.float32)
    vernier_labels = np.zeros_like(image_labels)  # 0 = left, 1 = right

    if print_shapes:
        print('loading ' + stim_type + ' from '+folder)

    for image in image_files:
        image_file = os.path.join(folder, image)
        # print('loading '+image_file)

        if stim_type in image_file:
            for rep in range(n_repeats):
                try:
                    image_data = ndimage.imread(image_file, mode='L').astype(float)
                    image_data = misc.imresize(image_data, resize_factor)

                    # pad to the right size if image is small than image_size (image will be in a random place)
                    if any(np.less(image_data.shape, image_size)):
                        pos_x = random.randint(0, max(0, image_size[1]-image_data.shape[1]))
                        pos_y = random.randint(0, max(0, image_size[0] - image_data.shape[0]))
                        padded = np.zeros(image_size, dtype=np.float32)
                        padded[pos_y:pos_y+image_data.shape[0], pos_x:pos_x+image_data.shape[1]] = image_data
                        image_data = padded

                    # crop out a random patch if the image is larger than image_size
                    if image_data.shape[0] > image_size[0]:
                        first_row = int((image_data.shape[0]-image_size[0])/2)
                        image_data = image_data[first_row:first_row+image_size[0], :]
                    if image_data.shape[1] > image_size[1]:
                        first_col = int((image_data.shape[1]-image_size[1])/2)
                        image_data = image_data[:, first_col:first_col+image_size[1]]

                    # normalize etc.
                    # zero mean, 1 stdev
                    image_data = (image_data - np.mean(image_data)) / np.std(image_data)

                    if random.randint(1, 100) > 50:
                        image_data = np.fliplr(image_data)
                        vernier_labels[num_images] = 1

                    # add to training set
                    image_batch[num_images, :, :] = image_data

                    if 'square' in image_file:
                        image_labels[num_images] = 0
                    elif 'circle' in image_file:
                        image_labels[num_images] = 1
                    elif 'hexagon' in image_file:
                        image_labels[num_images] = 2
                    elif 'octagon' in image_file:
                        image_labels[num_images] = 3
                    elif 'star' in image_file:
                        image_labels[num_images] = 4
                    elif 'line' in image_file:
                        image_labels[num_images] = 5
                    elif 'vernier' in image_file:
                        image_labels[num_images] = 6
                        # print('CAREFUL WITH LABELS WHEN DOING DIFFERENT TYPES OF TASK!')
                    else:
                        image_labels[num_images] = 7

                    num_images = num_images + 1

                except (ValueError, IOError, IndexError, OSError):
                    print('Could not read:', image_file, '- it\'s ok, skipping.')

    # check enough images could be processed
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    # remove empty entries
    image_batch = image_batch[0:num_images, :, :]
    image_labels = image_labels[0:num_images]
    vernier_labels = vernier_labels[0:num_images]
    # add a singleton 4th dimension (needed for conv layers
    image_batch = np.expand_dims(image_batch, axis=3)

    if print_shapes:
        print('Image batch tensor:', image_batch.shape)
        print('Mean:', np.mean(image_batch))
        print('Standard deviation:', np.std(image_batch))

    return image_batch, image_labels, vernier_labels
