from scipy.fft import fft2, ifft2

def fft_deconvolve(image, psf):

    # Ensure PSF has odd number of pixels because that's easier
    assert psf.shape[0] % 2 == 0, "psf image must have odd number of pixels"
    assert psf.shape[1] % 2 == 0, "psf image must have odd number of pixels"
    
    # ensure image has odd number of pixels because that's easier
    if np.any(image.shape % 2 == 0):
        print('WARNING: input image has even number of pixles, Im going to remove a pixel from that side.')
    image = image[0:image.shape[0] - 1 + (image.shape[0] % 2),
                  0:image.shape[1] - 1 + (image.shape[1] % 2)]

    image_fft = fft2(image)
    psf_fft = fft2(psf, image.shape)

    deconvolved_fft = image_fft / psf_fft

    return np.real(ifft(deconvolved_image))
    
