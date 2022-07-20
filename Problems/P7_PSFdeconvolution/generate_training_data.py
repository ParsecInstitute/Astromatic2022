import sys
import os
sys.path.insert(0, os.path.join(os.getenv("ASTROMATIC_PATH"), "Problems"))
import numpy as np
from utils.desi_image_downloader import get_image
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel, AiryDisk2DKernel
from astropy.io import fits
import time
import h5py
from scipy.stats import iqr
from tqdm import tqdm


# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!


def square_window(npix, aperture_frac=1.):
	window = np.zeros(npix)
	aperture = npix * aperture_frac
	cutout_side = np.arange(-npix / 2, npix / 2)
	r = np.abs(cutout_side)
	opening = r < aperture / 2
	window[opening == 1.0] = 1.
	window = np.outer(window, window)

	return window


class DataPicker():
	def __init__(self, npix, cuts_dataset, min_sum=10, sat_pixel=9, aperture_frac=1.):

		self.npix = npix
		self.min_sum = min_sum
		self.sat_pixel = sat_pixel
		self.aperture_frac = aperture_frac
		self.window = square_window(self.npix, self.aperture_frac)
		self.cuts_dataset = cuts_dataset
		self.cuts_id = 0

	def select(self):
		# timeouts = 0
		while True:
			# RA = np.random.uniform(low=RA_bounds[0], high=RA_bounds[1])
			# DEC = np.random.uniform(low=DEC_bounds[0], high=DEC_bounds[1])
			# try:
			# 	cutout, header = get_image(RA, DEC, size=self.npix)
			# except:
			# 	# timeouts += 1
			# 	continue
			hdul = fits.open(os.path.join(self.cuts_dataset, f"cut{self.cuts_id:05d}.fits"))
			self.cuts_id += 1
			cutout = hdul[0].data
			light_sum = np.sum(self.window * cutout)
			no_sat = np.all(cutout < self.sat_pixel)
			if light_sum > self.min_sum and no_sat:
				break
			print("bad cutout, retrying...")
		# print(f"{timeouts} timeouts")
		return cutout


class Corrupter():
	def __init__(self, npix, psf_npix, psf="gaussian", noise="gaussian"):

		self.npix = npix
		self.psf_npix = psf_npix
		self.psf = psf
		self.noise = noise

	def add_psf(self, img):
		if self.psf == "gaussian":
			width = np.random.uniform(low=psfwidth_bounds[0], high=psfwidth_bounds[1])
			self.radius = Corrupter.fwhm2std(width)
			kernel = Gaussian2DKernel(x_stddev=self.radius, x_size=self.psf_npix, y_size=self.psf_npix)
		elif self.psf == "airy":
			self.radius = np.random.uniform(low=psfwidth_bounds[0], high=psfwidth_bounds[1])
			kernel = AiryDisk2DKernel(radius=self.radius, x_size=self.psf_npix, y_size=self.psf_npix)
		else:
			raise ValueError(f"psf of type '{self.psf}' not implemented")
		# blurry_img = convolve(img, kernel)
		blurry_img = convolve_fft(img, kernel)
		return blurry_img, kernel.array
	def add_noise(self, img):
		if self.noise == "gaussian":
			scale = iqr(img, rng=(16,84))/2
			mask = np.random.normal(loc=0., scale=scale, size=(self.npix, self.npix))
		else:
			raise ValueError(f"noise of type '{self.noise}' not implemented")
		img += mask
		return img

	@staticmethod
	def fwhm2std(width):
		return width ** 2 / (8*np.log(2))


def produce_dataset(output_path, cuts_dataset_dir, set_size, wmode="w-", rpf=50, gen_params={}):
	"""
	Produce a dataset of corrupt DESI images
	:param output_path: path to dataset directory (str)
	:param set_size: number of examples in dataset to be produced (int)
	:param wmode: h5py write mode (str)
	:param rpf: number of realizations to save per file (int)
	:param gen_params: kwargs for generator function
	"""
	if rpf > set_size:
		raise ValueError(f"rpf cannot be larger than set_size")

	dataset_name = os.path.basename(output_path)
	if dataset_name == "":
		raise ValueError("Invalid output_path")

	if not os.path.isdir(output_path):
		os.makedirs(output_path, exist_ok=True)

	def basegrp_header(grp):
		header_dic = {"set_size": set_size,
					  "npix": gen_params["npix"],
					  "psf_npix": gen_params["psf_npix"],
					  "psf": gen_params["psf"],
					  "noise": gen_params["noise"],
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

	picker = DataPicker(gen_params["npix"], cuts_dataset_dir)
	corrupter = Corrupter(**gen_params)

	corrupt_dset = basegrp.create_dataset(f"corrupt", (rpf, gen_params["npix"], gen_params["npix"]), dtype=float, compression="gzip")
	psf_dset = basegrp.create_dataset(f"psf", (rpf, gen_params["psf_npix"], gen_params["psf_npix"]), dtype=float, compression="gzip")
	truth_dset = basegrp.create_dataset(f"truth", (rpf, gen_params["npix"], gen_params["npix"]), dtype=float, compression="gzip")

	for i, r in tqdm(enumerate(work_ids)):
		if i % rpf == 0 and i!= 0:
			output_file.close()
			ind_file += 1
			output_file = h5py.File(os.path.join(output_path, f"{dataset_name}_{ind_file:04d}.h5"), mode=wmode)
			basegrp = output_file.create_group("base")
			basegrp_header(basegrp)

			corrupt_dset = basegrp.create_dataset(f"corrupt", (rpf, gen_params["npix"], gen_params["npix"]),
												  dtype=float, compression="gzip")
			psf_dset = basegrp.create_dataset(f"psf", (rpf, gen_params["psf_npix"], gen_params["psf_npix"]), dtype=float,
											  compression="gzip")
			truth_dset = basegrp.create_dataset(f"truth", (rpf, gen_params["npix"], gen_params["npix"]), dtype=float,
												compression="gzip")

		# generation
		cut = picker.select()
		blurry_cut, kernel = corrupter.add_psf(cut)
		corrupt_cut = corrupter.add_noise(blurry_cut)

		corrupt_dset[i%rpf, ...] = corrupt_cut
		psf_dset[i%rpf, ...] = kernel
		truth_dset[i%rpf, ...] = cut

	prod_end = time.time()
	timer = prod_end - prod_start
	print(f"dataset produced in {int(timer // 60)} minutes {(timer % 60):.02f} seconds")


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(
		description="Produce a dataset of corrupt DESI cutouts and their associated PSF")

	parser.add_argument("--path_in", type=str, default=os.path.join(os.getenv("ASTROMATIC_PATH")),
						help="principal repo path")
	parser.add_argument("--path_out", type=str,
						default=os.path.join(os.getenv("ASTROMATIC_PATH"), "Problems", "P7_PSFdeconvolution",
											 "datasets"),
						help="output path")
	parser.add_argument("--rpf", type=int, default=50, help="number of realizations per file")
	parser.add_argument("--ow", action="store_const", const="w", default="w-",
						help="toggle to overwrite existing dataset in same path")
	parser.add_argument("--dataset_name", type=str, required=True, help="name of dataset directory")
	parser.add_argument("--cuts_dataset", type=str, required=True, help="name of dataset containing raw DESI cuts")
	parser.add_argument("--npix", type=int, default=256, help="number of pixels of selected cuts")
	parser.add_argument("--psf_npix", type=int, default=49, help="pixel size of psf, must be odd")
	parser.add_argument("--psf", type=str, default="gaussian",
						help="type of psf to corrupt images, in ['gaussian', 'airy']")
	parser.add_argument("--noise", type=str, default="gaussian",
						help="type of noise to corrupt images, in ['gaussian', 'poisson']")
	parser.add_argument("--size", type=int, required=True, help="size of dataset to produce")
	parser.add_argument("--seed", type=int, default=None, help="random seed")

	args = parser.parse_args()

	np.random.seed(args.seed)

	gen_params = {"npix": args.npix,
				  "psf_npix": args.psf_npix,
				  "noise": args.noise,
				  "psf": args.psf}

	# BOUNDS
	RA_bounds = (115, 260)
	DEC_bounds = (0, 80)
	psfwidth_bounds = (4, 8) 		# psf_npix of 49 is too narrow for around 8pix width

	wmode = args.ow
	dataset_dir = os.path.join(args.path_out, args.dataset_name)
	cuts_dataset_dir = os.path.join(args.path_out, args.cuts_dataset)
	produce_dataset(dataset_dir, cuts_dataset_dir, args.size, wmode, args.rpf, gen_params)