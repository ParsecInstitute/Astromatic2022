import numpy as np
from skimage.restoration import richardson_lucy


def deconvolve_lucyrichardson(image, psf, n_iter=100, filter_epsilon=None):
    """
    Deconvolve a PSF from an image and return the sharpened image.

    image: 2d numpy array image
    psf: 2d numpy image with centered point source. Must have odd number of pixels on each side. Point source is centered at the middle of the central pixel.
    n_iter: number of Lucy-Richardson iterations to perform.
    """

    # Ensure PSF has odd number of pixels because that's easier
    assert psf.shape[0] % 2 != 0, "psf image must have odd number of pixels"
    assert psf.shape[1] % 2 != 0, "psf image must have odd number of pixels"

    # Record pixel flux limits from image. These are used to scale to the -1,1 range
    dmax = np.max(image)
    dmin = np.min(image)

    # Perform the LR deconvolution on the scaled image
    deconv = richardson_lucy(
        2 * (image - dmin) / (dmax - dmin) - 1,
        psf,
        num_iter=n_iter,
		filter_epsilon=filter_epsilon
    )

    # Rescale back to the original flux range and return
    return (deconv + 1) * ((dmax - dmin) / 2) + dmin
