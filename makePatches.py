# Create vernier, crowding and uncrowding patches
import os, numpy as np, random, scipy.misc, matplotlib.pyplot as plt
from scipy import ndimage, misc


# Create vernier patch
def createVernierPatch(barHeight, barWidth, offsetHeight, offsetWidth, offsetDir):

    patch = np.zeros((2*barHeight+offsetHeight, 2*barWidth+offsetWidth))
    patch[0                     :barHeight, 0                   :barWidth] = 254.0
    patch[barHeight+offsetHeight:         , barWidth+offsetWidth:        ] = 254.0

    if offsetDir:
        patch = np.fliplr(patch)

    return patch

# Create square patch
def createSquarePatch(barHeight, barWidth, offsetHeight, offsetWidth, offsetDir):

    critDist  = int((barHeight+offsetHeight)/2)
    squareDim = max(2*barHeight+offsetHeight, 2*barWidth+offsetWidth)+2*critDist+2*barWidth
    patch     = np.zeros((squareDim, squareDim))
    patch[0:barWidth, :] = 254.0
    patch[-barWidth:, :] = 254.0
    patch[:, 0:barWidth] = 254.0
    patch[:, -barWidth:] = 254.0

    return patch

# Create crowded patch
def createCrowdedPatch(barHeight, barWidth, offsetHeight, offsetWidth, offsetDir):

    critDist        = int((barHeight+offsetHeight)/2)
    patch           = createSquarePatch( barHeight, barWidth, offsetHeight, offsetWidth, offsetDir)
    vernierPatch    = createVernierPatch(barHeight, barWidth, offsetHeight, offsetWidth, offsetDir)
    firstVernierRow = critDist+barWidth
    firstVernierCol = int(patch.shape[0]/2 - (barWidth+offsetWidth/2))

    patch[firstVernierRow:firstVernierRow+2*barHeight+offsetHeight, firstVernierCol:firstVernierCol+2*barWidth+offsetWidth] = vernierPatch

    return patch

# Create uncrowded patch (7 squares)
def createUncrowdedPatch(barHeight, barWidth, offsetHeight, offsetWidth, offsetDir):

    critDist     = int((barHeight+offsetHeight)/2)
    squarePatch  = createSquarePatch(barHeight, barWidth, offsetHeight, offsetWidth, offsetDir)
    crowdedPatch = createCrowdedPatch(barHeight, barWidth, offsetHeight, offsetWidth, offsetDir)
    oneSquareDim = squarePatch.shape[0]
    patch        = np.zeros((oneSquareDim, 7*oneSquareDim + 6*critDist))

    for n in range(7):
        firstCol = n*(oneSquareDim+critDist)
        if n == 3:
            patch[:, firstCol:firstCol+oneSquareDim] = crowdedPatch
        else:
            patch[:, firstCol:firstCol+oneSquareDim] = squarePatch

    return patch

# Main function
def createPatches(nSamples, stimType, im_size=(60,128)):

    if not os.path.exists(stimType):
        os.mkdir(stimType)

    barHeightRange    = range(5, 20)
    barWidthRange     = range(2,  5)
    offsetHeightRange = range(0,  5)
    offsetWidthRange  = range(2,  7)
    names             = ['L', 'R']

    for offset in (0,1):
        name = stimType+'/'+names[offset]
        n    = 0
        while n < nSamples:

            bH = random.choice(barHeightRange)
            bW = random.choice(barWidthRange)
            oH = random.choice(offsetHeightRange)
            oW = random.choice(offsetWidthRange)

            if stimType   == 'vernier':
                thisPatch = np.pad(createVernierPatch(  bH, bW, oH, oW, offset), 10, mode='constant')
            elif stimType == 'crowded':
                thisPatch = np.pad(createCrowdedPatch(  bH, bW, oH, oW, offset), 10, mode='constant')
            elif stimType == 'uncrowded':
                thisPatch = np.pad(createUncrowdedPatch(bH, bW, oH, oW, offset), 10, mode='constant')
            else:
                raise Exception('Unknown stimulus type.')

            if thisPatch.shape[0] > im_size[0] or thisPatch.shape[1] > im_size[1]:
                pass
            else:
                n += 1
                thisName = name+str(n)
                # np.save(thisName, np.array([thisPatch, thisPatch, thisPatch]))
                scipy.misc.imsave(thisName+'.png', thisPatch) # np.array([thisPatch, thisPatch, thisPatch]))

                # Example main call
                #createPatches(10, 'vernier')
                #createPatches(10, 'crowded')
                #createPatches(10, 'uncrowded')

                # plt.figure()
                # plt.imshow(thisPatch, interpolation='nearest')
                # # ax = plt.gca()
                # # ax.set_xticks(np.arange(0, thisPatch.shape[1], 1));
                # # ax.set_yticks(np.arange(0, thisPatch.shape[0], 1));
                # # ax.set_xticklabels(np.arange(1, thisPatch.shape[0]+1, 1));
                # # ax.set_yticklabels(np.arange(1, thisPatch.shape[0]+1, 1));
                # # ax.set_xticks(np.arange(-.5, thisPatch.shape[1], 1), minor=True);
                # # ax.set_yticks(np.arange(-.5, thisPatch.shape[1], 1), minor=True);
                # # ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                # plt.show()


# input patches should be n_patches x im_size and labels should be a n_patches vector of 0s and 1s
def make_dataset_from_patch(patch_folder, image_size=(60, 128), resize_factor=1.0, n_repeats=10,
                            print_shapes=False):
    patch_folder = patch_folder + '/'

    min_num_images = 1
    num_images = 0

    image_files = os.listdir(patch_folder)

    data = np.ndarray(shape=(n_repeats * len(image_files), image_size[0], image_size[1], 3),
                      dtype=np.float32)
    labels = np.zeros(n_repeats * len(image_files), dtype=np.float32)

    print('loading images from ' + patch_folder)

    for image in image_files:

        image_file = os.path.join(patch_folder, image)

        for rep in range(n_repeats):
            try:

                this_image = ndimage.imread(image_file, mode='L').astype(float)
                this_image = misc.imresize(this_image, resize_factor)
                this_image = np.expand_dims(this_image, -1)

                # crop out a random patch if the image is larger than image_size
                if this_image.shape[0] > image_size[0]:
                    firstRow = int((this_image.shape[0] - image_size[0]) / 2)
                    this_image = this_image[firstRow:firstRow + image_size[0], :, :]
                if this_image.shape[1] > image_size[1]:
                    firstCol = int((this_image.shape[1] - image_size[1]) / 2)
                    this_image = this_image[:, firstCol:firstCol + image_size[1], :]

                # pad to the right size if image is small than image_size (image will be in a random place)
                if any(np.less(this_image.shape, image_size)):
                    posX = random.randint(0, max(0, image_size[1] - this_image.shape[1]))
                    posY = random.randint(0, max(0, image_size[0] - this_image.shape[0]))
                    padded = np.zeros(image_size, dtype=np.float32)
                    padded[posY:posY + this_image.shape[0], posX:posX + this_image.shape[1], :] = this_image
                    this_image = padded

                # normalize etc.
                # zero mean, 1 stdev
                this_image = (this_image - np.mean(this_image)) / np.std(this_image)

                # add to dataset
                data[num_images, :, :, :] = this_image

                if 'L' in image_file:
                    labels[num_images] = 0
                elif 'R' in image_file:
                    labels[num_images] = 1
                else:
                    raise Exception(image_file + ' is a stimulus of unknown class')

                num_images = num_images + 1

            except ():
                print('Could make image from patch:', image_file, '- it\'s ok, skipping.')

    # check enough images could be processed
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    # remove empty entries
    data = data[0:num_images, :, :, :]

    perm = np.random.permutation(num_images)
    data = data[perm, :, :, :]
    labels = labels[perm]

    if print_shapes:
        print('Dataset tensor:', data.shape)
        print('Mean:', np.mean(data))
        print('Standard deviation:', np.std(data))

    return data, labels
