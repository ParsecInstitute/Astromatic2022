from scipy.fft import fft2, ifft2


def deconvolve_fft(image, psf):
    """
    Deconvolve a PSF from an image and return the sharpened image.

    image: 2d numpy array image
    psf: 2d numpy image with centered point source. Must have odd number of pixels on each side. Point source is centered at the middle of the central pixel.
    """

    # Ensure PSF has odd number of pixels because that's easier
    assert psf.shape[0] % 2 == 0, "psf image must have odd number of pixels"
    assert psf.shape[1] % 2 == 0, "psf image must have odd number of pixels"

    # ensure image has odd number of pixels because that's easier
    if np.any(image.shape % 2 == 0):
        print(
            "WARNING: input image has even number of pixles, Im going to remove a pixel from that side."
        )
    image = image[
        0 : image.shape[0] - 1 + (image.shape[0] % 2),
        0 : image.shape[1] - 1 + (image.shape[1] % 2),
    ]

    # Convert image and psf to frequency space
    image_fft = fft2(image)
    psf_fft = fft2(psf, image.shape)

    # Deconvolution operation is division in frequency space
    deconvolved_fft = image_fft / psf_fft

    # Return real space deconvolved image
    return np.real(ifft(deconvolved_image))
