from copy import deepcopy
from astropy.io import fits

params = {
    "DESI": {'layer': 'ls-dr9', 'pixelscale': 0.262, 'band': 'r', 'size': 3000},
    'unWISE': {'layer': 'unwise-neo6', 'pixelscale': 2.75, 'band': 'w1', 'size': 1000},
    'GALEX': {'layer': 'galex', 'pixelscale': 1.5, 'band': 'n', 'size': 1000}
}

request_url = "https://www.legacysurvey.org/viewer/fits-cutout?ra={RA}&dec={DEC}&size={size}&layer={layer}&pixscale={pixelscale}&bands={band}"

def get_image(RA, DEC, survey = 'DESI', return_full_fits = False, **kwargs):
    """
    Given coordiantes, downloads a lagacy survey image and returns the information. 

    RA: Right Ascension [float, deg]
    DEC: Declination [float, deg]
    survey: Survey image group, choose from DESI, unWISE, and GALEX. [str]
    return_full_fits: boolean to return the full fits object, or return the ndarray image and header string [bool]
    kwargs: options to override the survey parameters, can be layer [str], pixelscale [float], band [str], or size [int].

    returns: astropy fits object, or ndarray image and header string
    """
    request_kwargs = deepcopy(params[survey])
    request_kwargs.update(kwargs)
    url = request_url.format(RA=RA, DEC=DEC, **request_kwargs)
    hdul = fits.open(url)
    
    if return_full_fits:
        return hdul
    else:
        return hdul[0].data, hdul[0].header
