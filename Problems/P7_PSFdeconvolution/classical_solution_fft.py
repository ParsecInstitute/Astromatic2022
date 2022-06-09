import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

def deconvolve_fft(image, psf, pad_size = 100):
    """
    Deconvolve a PSF from an image and return the sharpened image.

    image: 2d numpy array image
    psf: 2d numpy image with centered point source. Must have odd number of pixels on each side. Point source is centered at the middle of the central pixel.
    """

    # Ensure PSF has odd number of pixels because that's easier
    assert psf.shape[0] % 2 != 0, "psf image must have odd number of pixels"
    assert psf.shape[1] % 2 != 0, "psf image must have odd number of pixels"
    
    image = np.pad(image, pad_width = pad_size, mode = 'edge')
    
    # Convert image and psf to frequency space
    image_fft = fftshift(fft2(image))
    psf_fft = fftshift(fft2(psf, image.shape)) 
    # Deconvolution operation is division in frequency space
    deconvolved_fft = image_fft / psf_fft
    cut_freq = int(image.shape[0]/2 - 5*psf.shape[0])
    smooth_fft = np.zeros(deconvolved_fft.shape)
    smooth_fft[cut_freq:-cut_freq,cut_freq:-cut_freq] = deconvolved_fft[cut_freq:-cut_freq,cut_freq:-cut_freq]
    # Return real space deconvolved image
    deconvolved_image = np.abs(ifft2(ifftshift(smooth_fft)))

    return deconvolved_image[pad_size:-pad_size,pad_size:-pad_size]
    
