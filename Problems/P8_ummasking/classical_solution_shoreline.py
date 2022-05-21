import numpy as np
from astropy.convolution import convolve

def unmask_shoreline(image, mask, kernel_size = 5):
    """
    Fills in masked areas by taking a local average. Will only fill pixels that are immediately adjacent to unmasked pixels. Iteratively fills in "shoreline" pixels until the entire image has been filled.

    image: 2d numpy array image
    mask: 2d numpy boolean array with 1 or True indicating a pixel is masked
    kernel_size: region within which to take averages for filling pixels

    returns: 2d numpy array with all masked values filled
    """

    # make the masked numpy array
    mimage = np.ma.masked_array(image, mask = mask)

    # loop until all masked pixels are filled
    while np.any(mimage.mask):

        # Compute the distances to find masked pixels that are adjacent to real values
        DD = distance_transform_edt(mimage.mask == 1, return_distances = True, return_indices = False)

        # Apply the averaging kernel to get fill values
        smooth = convolve(mimage, np.ones((kernel_size, kernel_size))/kernel_size**2, boundary = 'extend')

        # Replace only the "shoreline" pixels with the local average
        mimage[DD == 1] = smooth[DD == 1]

    return mimage.data

