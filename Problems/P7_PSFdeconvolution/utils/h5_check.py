import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from display_utils import display



filepath = os.path.join(os.getenv("ASTROMATIC_PATH"), "Problems", "P7_PSFdeconvolution", "datasets", "debug_deconv_dataset", "debug_deconv_dataset_0000.h5")

file = h5py.File(filepath, mode="r")

desc = file["base"].attrs["dataset_descriptor"]

file.close()