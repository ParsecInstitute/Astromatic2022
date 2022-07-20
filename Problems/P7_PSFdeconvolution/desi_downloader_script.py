import sys
sys.path.insert(0, "../")
import numpy as np
import os
from utils.desi_image_downloader import get_image
from astropy.io import fits



if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(
		description="Download a given number of DESI cutouts of a certain size")

	parser.add_argument("--path_out", type=str,
						default=os.path.join(os.getenv("ASTROMATIC_PATH"), "Problems", "P7_PSFdeconvolution",
											 "datasets"),
						help="output path")
	parser.add_argument("--dataset_name", type=str, required=True, help="name of dataset directory")
	parser.add_argument("--npix", type=int, default=1001, help="number of pixels of selected cuts")
	parser.add_argument("--size", type=int, required=True, help="size of dataset to produce")
	parser.add_argument("--seed", type=int, default=None, help="random seed")

	args = parser.parse_args()

	np.random.seed(args.seed)

	# BOUNDS
	RA_bounds = (115, 260)
	DEC_bounds = (0, 80)

	dataset_dir = os.path.join(args.path_out, args.dataset_name)
	if not os.path.exists(dataset_dir):
		os.makedirs(dataset_dir, exist_ok=True)


	for i in range(args.size):
		RA = np.random.uniform(low=RA_bounds[0], high=RA_bounds[1])
		DEC = np.random.uniform(low=DEC_bounds[0], high=DEC_bounds[1])

		hdul = get_image(RA, DEC, return_full_fits=True, size=args.npix)

		hdul.writeto(os.path.join(dataset_dir, f"cut{i:05d}.fits"), overwrite=True)