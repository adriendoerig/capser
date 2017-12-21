import numpy as np

# Taken from: https://github.com/tensorflow/tensorflow/issues/6322
def images_to_sprite(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""

    if isinstance(images, list):
        images = np.array(images)

    # if images are not square, make them square
    if images.shape[1] != images.shape[2]:
       # print('Images are not square: squaring them for the sprite image')
        square_images = np.zeros((images.shape[0],max(images[0,:,:].shape),max(images[0,:,:].shape)))
        for im in range(images.shape[0]):
            square_images[im,:images.shape[1],:images.shape[2]] = images[im,:,:]
        images = square_images

    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = this_img

    return spriteimage

def invert_grayscale(input_image):
    """ Makes black white, and white black """
    return 1 - input_image

