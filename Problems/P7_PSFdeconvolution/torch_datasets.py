import os
from glob import glob
import time
import torch
from torch.utils.data import Dataset
import ast
import h5py
import numpy as np
from skimage.transform import resize


class PSFDataset(Dataset):
	"""
	pyTorch Dataset for loading corrupt images and PSFs and their associated original images.
	Assumes identical shape for all three
	"""
	def __init__(self, dir_path, method="train", val_frac=0.2, test_frac=0.2, size=None):

		self.size = size
		self.dir_path = dir_path
		self.file_list = sorted(glob(f"{self.dir_path}/*.h5"))

		# Only open one file to get descriptor
		for i, file in enumerate(self.file_list):
			while True:
				try:
					with h5py.File(file, mode="r") as h5_file:
						desc = ast.literal_eval(h5_file["base"].attrs["dataset_descriptor"])
						self.set_size = desc["set_size"]
						self.npix = desc["npix"]
						self.rpf = desc["rpf"]

				except:
					time.sleep(1.)
					continue
				else:
					break

		self.all_ids = np.arange(self.set_size)

		if self.size is None: self.size = self.set_size
		self.all_ids = np.random.choice(self.all_ids, self.size, replace=False)

		arg1 = int((1 - val_frac - test_frac) * self.size)
		arg2 = int((1 - test_frac) * self.size)
		self.index = 0
		self.ids = {
			"train": self.all_ids[0:arg1],
			"val": self.all_ids[arg1:arg2],
			"test": self.all_ids[arg2:]
		}

		self.method = method

	def __len__(self):
		return len(self.ids[self.method])

	def __getitem__(self, idx):
		"""
		get single sample from dataset
		"""
		sample_id = self.ids[self.method][idx]
		sample_file = self.file_list[sample_id%self.rpf]

		element_id = sample_id//self.rpf

		# algorithm to avoid hypy IOError when many workers read same file
		while True:
			try:
				with h5py.File(sample_file, mode="r") as h5_file:
					corrupt_arr = h5_file["base"]["corrupt"][element_id]
					psf_arr = h5_file["base"]["psf"][element_id]
					truth_arr = h5_file["base"]["truth"][element_id]
			except:
				time.sleep(1.)
				continue
			else:
				break

		corrupt_tensor = torch.tensor(corrupt_arr).reshape(1, self.npix, self.npix)
		psf_tensor = torch.tensor(psf_arr).reshape(1, self.npix, self.npix)
		y = torch.tensor(truth_arr).reshape(1, self.npix, self.npix)

		x = torch.cat((corrupt_tensor, psf_tensor), 0)

		return x.float(), y.float()