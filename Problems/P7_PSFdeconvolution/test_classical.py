import sys
sys.path.append('utils/')
from desi_image_downloader import get_image
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import SqrtStretch, LogStretch, HistEqStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from scipy.stats import iqr
from astropy.convolution import convolve
from classical_solution_fft import deconvolve_fft
from classical_solution_lucyrichardson import deconvolve_lucyrichardson
sigma = 4
IMG, header = get_image(140, 12, size = 1001)
noise = iqr(IMG, rng = (16, 84))/2
plt.imshow(IMG, origin = "lower",
           norm = ImageNormalize(stretch=HistEqStretch(IMG), clip=False))
plt.show()
XX, YY = np.meshgrid(np.arange(25) - 12, np.arange(25) - 12)
RR = np.sqrt(XX**2 + YY**2)
kernel = np.exp(-0.5 * (RR / sigma)**2)
kernel /= np.sum(kernel)
conv_IMG = convolve(IMG, kernel)
plt.imshow(conv_IMG, origin = "lower",
           norm = ImageNormalize(stretch=HistEqStretch(conv_IMG), clip=False))
plt.show()
deconv_fft_IMG = deconvolve_fft(conv_IMG, kernel)
plt.imshow(deconv_fft_IMG, origin = "lower",
           norm = ImageNormalize(stretch=HistEqStretch(deconv_fft_IMG), clip=False))
plt.show()
deconv_LR_IMG = deconvolve_lucyrichardson(conv_IMG, kernel)
plt.imshow(deconv_LR_IMG, origin = "lower",
           norm = ImageNormalize(stretch=HistEqStretch(deconv_LR_IMG), clip=False))
plt.show()
