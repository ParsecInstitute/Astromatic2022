import os
from glob import glob
import numpy as np
from numpy.random import uniform as unif
import matplotlib.pyplot as plt
from matplotlib import gridspec
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from astropy.cosmology import Planck18 as cosmo
from astropy.convolution import convolve, Gaussian2DKernel
from astromatic.lens_inference.utils.lensing_utils import theta_E_from_M, sp_ray_tracing, lens_source
from astromatic.lens_inference.utils.display_utils import display_diff
import scipy.interpolate as sp_interp
import h5py
from tqdm import tqdm


class GalaxyLenser():
	"""
	Class for the lensing of a single Sersic source galaxy by an SIE + SHEAR of parameters randomly sampled parameters
	"""
	def __init__(self, shear=True, noise="poisson", psf="gaussian", mass_function="beta", mass_function_kw={"a": 5, "b": 2.5}, include_lens_light=False):

		self.shear = shear
		self.noise = noise
		self.psf = psf
		self.mass_function = mass_function
		self.mass_function_kw = mass_function_kw
		self.include_lens_light = include_lens_light

		self.sample_mass()
		self.lens_paramsampler()
		self.source_paramsampler()
		if self.include_lens_light:
			self.lens_light_paramsampler()

		self.format_params()

	def sample_mass(self, log_mlow=10.7, log_mhigh=12.):

		if self.mass_function == "beta":
			sample = np.random.beta(**self.mass_function_kw)

		self.log_lens_mass = GalaxyLenser._min_max_scale(sample, log_mlow, log_mhigh)
		self.lens_mass = 10 ** self.log_lens_mass


	def lens_paramsampler(self):
		if not hasattr(self, "lens_mass"):
			self.sample_mass()

		# SIE (+ SHEAR) lens
		SIE_kwargs = {"theta_E": theta_E_from_M(self.lens_mass, ZL, ZS),
					  "e1": unif(low=SIE_pb["e1"][0], high=SIE_pb["e1"][1]),
					  "e2": unif(low=SIE_pb["e2"][0], high=SIE_pb["e2"][1]),
					  "center_x": unif(low=SIE_pb["center_x"][0], high=SIE_pb["center_x"][1]),
					  "center_y": unif(low=SIE_pb["center_y"][0], high=SIE_pb["center_y"][1])}

		self.lens_kwargs = [SIE_kwargs]
		self.lens_redshift_list = [ZL]
		self.lens_model_list = ["SIE"]

		if self.shear:
			SHEAR_kwargs = {"gamma1": unif(low=SHEAR_pb["gamma1"][0], high=SHEAR_pb["gamma1"][1]),
							"gamma2": unif(low=SHEAR_pb["gamma2"][0], high=SHEAR_pb["gamma2"][1]),
							"ra_0": unif(low=SHEAR_pb["ra_0"][0], high=SHEAR_pb["ra_0"][1]),
							"dec_0": unif(low=SHEAR_pb["dec_0"][0], high=SHEAR_pb["dec_0"][1])}

			self.lens_kwargs.append(SHEAR_kwargs)
			self.lens_redshift_list.append(ZL)
			self.lens_model_list.append("SHEAR")


	def source_paramsampler(self):
		# Sersic Ellise source
		SERSIC_E_kwargs = {"amp": unif(low=SERSIC_E_pb["amp"][0], high=SERSIC_E_pb["amp"][1]),
					  	 "R_sersic": unif(low=SERSIC_E_pb["R_sersic"][0], high=SERSIC_E_pb["R_sersic"][1]),
					  	 "n_sersic": unif(low=SERSIC_E_pb["n_sersic"][0], high=SERSIC_E_pb["n_sersic"][1]),
						 "e1": unif(low=SERSIC_E_pb["e1"][0], high=SERSIC_E_pb["e1"][1]),
						 "e2": unif(low=SERSIC_E_pb["e2"][0], high=SERSIC_E_pb["e2"][1]),
						 "center_x": unif(low=SERSIC_E_pb["center_x"][0], high=SERSIC_E_pb["center_y"][1]),
						 "center_y": unif(low=SERSIC_E_pb["center_y"][0], high=SERSIC_E_pb["center_y"][1])}

		self.source_kwargs = [SERSIC_E_kwargs]
		self.source_redshift = ZS
		self.source_model_list = ["SERSIC_ELLIPSE"]


	def lens_light_paramsampler(self):
		if not hasattr(self, "lens_kwargs"):
			self.lens_paramsampler()

		# Sersic Ellise lens light
		SERSIC_E_kwargs = {"amp": unif(low=SERSIC_E_lens_pb["amp"][0], high=SERSIC_E_lens_pb["amp"][1]),
					  	 "R_sersic": unif(low=SERSIC_E_lens_pb["R_sersic"][0], high=SERSIC_E_lens_pb["R_sersic"][1]),
					  	 "n_sersic": unif(low=SERSIC_E_lens_pb["n_sersic"][0], high=SERSIC_E_lens_pb["n_sersic"][1]),
						 "e1": self.lens_kwargs[0]["e1"] + unif(-d_e_lim, d_e_lim),
						 "e2": self.lens_kwargs[0]["e2"] + unif(-d_e_lim, d_e_lim),
						 "center_x": self.lens_kwargs[0]["center_x"] + unif(-d_center_lim, d_center_lim),
						 "center_y": self.lens_kwargs[0]["center_y"] + unif(-d_center_lim, d_center_lim)}

		self.lens_light_kwargs = [SERSIC_E_kwargs]
		self.lens_light_redshift = ZL
		self.lens_light_model_list = ["SERSIC_ELLIPSE"]


	def produce_lens(self):
		# Source light
		self.source_light_model = LightModel(self.source_model_list)
		self.source_light = self.source_light_model.surface_brightness(BETA1, BETA2, self.source_kwargs)

		# Raytracing and lensing
		self.lens_model = LensModel(lens_model_list=self.lens_model_list,
									z_source=ZS,
									lens_redshift_list=self.lens_redshift_list)

		self.beta1_def, self.beta2_def = self.lens_model.ray_shooting(THETA1, THETA2, self.lens_kwargs)
		self.lensed_src = lens_source(THETA1, THETA2, self.beta1_def, self.beta2_def, self.source_light, NPIX)

		if self.include_lens_light:
			# Lens light
			self.lens_light_model = LightModel(self.lens_light_model_list)
			self.lens_light = self.lens_light_model.surface_brightness(THETA1, THETA2, self.lens_light_kwargs)

			self.lensed_image = self.lensed_src + self.lens_light
		else:
			self.lensed_image = self.lensed_src


	def corrupt_image(self, image):

		# PSF first
		if self.psf is not None:
			if self.psf == "gaussian":
				kernel = Gaussian2DKernel(x_stddev=2)
			image = convolve(image, kernel)

		# Noise
		if self.noise is not None:
			if self.noise == "poisson":
				mask = np.random.poisson(image)
			image += mask

		return image


	def format_params(self):
		self.lens_param_values = list(self.lens_kwargs[0].values())
		self.lens_param_keys = list(self.lens_kwargs[0].keys())

		if self.shear:
			self.lens_param_values += list(self.lens_kwargs[1].values())
			self.lens_param_keys += list(self.lens_kwargs[1].keys())

		self.source_param_values = list(self.source_kwargs[0].values())
		self.source_param_keys = list(self.source_kwargs[0].keys())

		self.lens_light_param_values = list(self.lens_light_kwargs[0].values())
		self.lens_light_param_keys = list(self.lens_light_kwargs[0].keys())


	@staticmethod
	def _min_max_scale(x, a, b):
		return a + x*(b-a)


def produce_dataset(output_path, set_size, wmode="w-", rpf=50, gen_params={}):
	"""
	Produce a dataset of lensed galaxy image realizations
	:param output_path: path to dataset directory (str)
	:param set_size: number of examples in dataset to be produced (int)
	:param mode: specifier for type of dataset to produce (str)
	:param wmode: h5py write mode (str)
	:param rpf: number of realizations to save per file (int)
	:param gen_params: kwargs for generator function
	"""
	dataset_name = os.path.basename(output_path)
	if dataset_name == "":
		raise ValueError("Invalid output_path")

	if not os.path.isdir(output_path):
		os.makedirs(output_path, exist_ok=True)

	def basegrp_header(grp):
		header_dic = {"set_size": set_size,
					  "ZL": gen_params["ZL"],
					  "ZS": gen_params["ZS"],
					  "LENS_FOV": gen_params["LENS_FOV"],
					  "SRC_FOV": gen_params["SRC_FOV"],
					  "lens_model": gen_params["lens_model"],
					  "src_model": gen_params["src_model"],
					  "data_type": gen_params["data_type"],
					  "rpf": rpf}

		grp.attrs["dataset_descriptor"] = str(header_dic)


	# SLURM array job separation
	real_ids = np.arange(set_size)
	rpw = set_size // N_WORKERS			# realizations per worker
	work_ids = real_ids[THIS_WORKER * rpw : (THIS_WORKER+1) * rpw]
	if THIS_WORKER + 1 == N_WORKERS:
		work_ids = real_ids[THIS_WORKER * rpw:]

	prod_start = time.time()
	ind_file = (rpw // rpf) * THIS_WORKER
	output_file = h5py.File(os.path.join(output_path, f"{dataset_name}_{ind_file:04d}.h5"), mode=wmode)
	basegrp = output_file.create_group("base")
	basegrp_header(basegrp)

	for i, r in enumerate(work_ids):
		if i % rpf == 0 and i!= 0:
			output_file.close()
			ind_file += 1
			output_file = h5py.File(os.path.join(output_path, f"{dataset_name}_{ind_file:04d}.h5"), mode=wmode)
			basegrp = output_file.create_group("base")
			basegrp_header(basegrp)

		# generation
		field = field_generator(**gen_params)
		alpha1_eff, alpha2_eff, alpha1_lp, alpha2_lp = GT_generator(field, **keyword_parse_GTgen(gen_params))

		# group for single realization
		grp_r = basegrp.create_group(f"real_{r:05d}")

		# NFW quantities
		grp_r.create_dataset(f"NFWredshifts_{r:05d}", data=field[f"{gen_params['skey']}redshifts"], dtype=float)
		arr_NFWparams = np.empty(shape=(4, field[f"n_{gen_params['skey'][:-1]}"]))
		for i, dic in enumerate(field[f"{gen_params['skey']}kwargs"]):
			arr_NFWparams[:, i] = list(dic.values())

		grp_r.create_dataset(f"NFWparams_{r:05d}", data=arr_NFWparams, dtype=float)

		# CONV quantities
		grp_r.create_dataset(f"CONVredshifts_{r:05d}", data=field["CONV_redshifts"], dtype=float)
		arr_CONVparams = np.empty(field["n_CONV"])			# we do not keep the convergence sheet's ra_0 and dec_0 because they are always (?) centered
		for i, dic in enumerate(field[f"CONV_kwargs"]):
			arr_CONVparams[i] = dic["kappa"]

		grp_r.create_dataset(f"CONVparams_{r:05d}", data=arr_CONVparams, dtype=float)

		# encode ground truth
		a1_eff_dset = grp_r.create_dataset(f"GTa1_eff_{r:05d}", (gen_params["NPIX"], gen_params["NPIX"]), dtype=float, compression="gzip")
		a1_eff_dset[...] = alpha1_eff

		a2_eff_dset = grp_r.create_dataset(f"GTa2_eff_{r:05d}", (gen_params["NPIX"], gen_params["NPIX"]), dtype=float, compression="gzip")
		a2_eff_dset[...] = alpha2_eff

		a1_lp_dset = grp_r.create_dataset(f"GTa1_lp_{r:05d}", (gen_params["NPIX"], gen_params["NPIX"]), dtype=float, compression="gzip")
		a1_lp_dset[...] = alpha1_lp

		a2_lp_dset = grp_r.create_dataset(f"GTa2_lp_{r:05d}", (gen_params["NPIX"], gen_params["NPIX"]), dtype=float, compression="gzip")
		a2_lp_dset[...] = alpha2_lp

	prod_end = time.time()
	timer = prod_end - prod_start
	print(f"dataset produced in {int(timer // 60)} minutes {(timer % 60):.02f} seconds")


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(
		description="Produce a dataset of strong lens images with or without lens light")

	parser.add_argument("--path_in", type=str, default=os.path.join(os.getenv("ASTROMATIC_PATH")),
						help="principal repo path")
	parser.add_argument("--path_out", type=str,	default=os.path.join(os.getenv("ASTROMATIC_PATH"), "astromatic", "lens_inference", "datasets"),
						help="output path")
	parser.add_argument("--dataset", type=str, default="lenses", help="name of dataset directory")
	parser.add_argument("--data_type", type=str, default="lens", help="type of data, in ['lens', 'lens_light']")
	parser.add_argument("--zl", type=float, default="0.5", help="redshift of lens")
	parser.add_argument("--zs", type=float, default="1.0", help="redshift of source")
	parser.add_argument("--npix", type=int, default=128, help="number of pixels at which to create data")
	parser.add_argument("--pixel_scale", type=float, default=0.05, help="scale of a single pixel in arcsecs")

	parser.add_argument("--noise", type=str, default="poisson", help="type of noise to corrupt images, in ['poisson']")
	parser.add_argument("--psf", type=str, default="gaussian", help="type of psf to corrupt images, in ['gaussian']")
	parser.add_argument("--size", type=int, default=None, help="size of dataset to produce")
	parser.add_argument("--seed", type=int, default=None, help="random seed")

	args = parser.parse_args()

	np.random.seed(args.seed)


	gen_params = {"NPIX": args.npix,
				  "ZL": args.zl,
				  "ZS": args.zs,
				  "pixel_scale": args.pixel_scale,
				  "LENS_FOV": args.npix * args.pixel_scale,
				  "SRC_FOV": args.npix * args.pixel_scale / 2,
				  ""}

	# --- CONSTANTS ----
	ZL = 0.5
	ZS = 1.0
	NPIX = args.npix
	pixel_scale = 0.05
	LENS_FOV = NPIX * pixel_scale
	SRC_FOV = 3

	# source plane coordinates
	src_grid_side = np.linspace(-SRC_FOV / 2, SRC_FOV / 2, NPIX)
	BETA1, BETA2 = np.meshgrid(src_grid_side, src_grid_side)

	# lens plane coordinates
	lens_grid_side = np.linspace(-LENS_FOV / 2, LENS_FOV / 2, NPIX)
	THETA1, THETA2 = np.meshgrid(lens_grid_side, lens_grid_side)

	# Lens model parambounds
	SIE_pb = {"e1": (-0.3, 0.3),  # prior bounds on theta_E defined from log mass bounds in GalaxyLenser.sample_mass
			  "e2": (-0.3, 0.3),
			  "center_x": (-0.15, 0.15),
			  "center_y": (-0.15, 0.15)}

	gamma_fac = 0.05
	SHEAR_pb = {"gamma1": (-gamma_fac, gamma_fac),
				"gamma2": (-gamma_fac, gamma_fac),
				"ra_0": (0, 0),
				"dec_0": (0, 0)}

	# Source model parambounds
	SERSIC_E_pb = {"amp": (10, 15),
				   "R_sersic": (0.05, 0.4),
				   "n_sersic": (1, 4),
				   "e1": (-0.4, 0.4),
				   "e2": (-0.4, 0.4),
				   "center_x": (-0.1, 0.1),
				   "center_y": (-0.1, 0.1)}

	# Lens light model priorbounds
	SERSIC_E_lens_pb = {"amp": (50, 60),
				   "R_sersic": (0.7, 1.2),
				   "n_sersic": (1, 2),
				   "e1": (-0.4, 0.4),
				   "e2": (-0.4, 0.4),
				   "center_x": (-0.1, 0.1),
				   "center_y": (-0.1, 0.1)}


	# Modifiers for lens light
	d_e_lim = 0.01
	d_center_lim = 0.01

	if args.data_type == "lens":
		include_lens_light = False
	elif args.data_type == "lens_light":
		include_lens_light = True


	# Production
	fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(21,20))

	for i in range(16):
		GL = GalaxyLenser(noise=args.noise, psf=args.psf, include_lens_light=include_lens_light)
		GL.produce_lens()
		GL.corrupt_image()

		display_diff(GL.lensed_image, ax[i//4, i%4], cmap="bone", lim=LENS_FOV/2)

	plt.tight_layout()
	plt.savefig("./lens_light_samples.png")
	plt.show()

	# display_diff(GL.source_light, cmap="hot")
	# display_diff(GL.lensed_src, cmap="hot")
	# display_diff(GL.lensed_image, cmap="hot")

	# print(GL.source_kwargs)
	# print(GL.lens_kwargs)
	# print(GL.lens_light_kwargs)

	print("debug")
