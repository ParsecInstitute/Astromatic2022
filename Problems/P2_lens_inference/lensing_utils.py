import numpy as np
from astropy import constants as csts
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
import scipy.interpolate as interp


def D_i_j(z_i, z_j):
	"""
    Angular diameter distance in Mpc between redshifts z_i and z_j
    with z_i < z_j
    """
	return cosmo.angular_diameter_distance_z1z2(z_i, z_j).value


def theta_E_from_M(M, zl, zs):
	"""
    Einstein radius in arcsecs of a mass M [Msun] at redshift zl, with source at zs
    """
	mass = M * u.Msun
	Dls = D_i_j(zl, zs) * u.Mpc
	Dl = D_i_j(0, zl) * u.Mpc
	Ds = D_i_j(0, zs) * u.Mpc

	theta_E = (np.sqrt(4 * csts.G * mass / csts.c ** 2 * Dls / Dl / Ds)).to(
		u.dimensionless_unscaled) * u.rad

	return theta_E.to(u.arcsec).value


def sp_ray_tracing(x1, x2, a1, a2):
	y1 = x1 - a1
	y2 = x2 - a2
	return y1, y2


def lens_source(x1, x2, y1, y2, source, npix):
	im = interp.griddata(points=(x1.ravel(), x2.ravel()), values=source.ravel(), xi=(y1.ravel(), y2.ravel()), method="linear", fill_value=0.)
	return im.reshape(npix, npix)
