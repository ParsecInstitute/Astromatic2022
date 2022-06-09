import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import SqrtStretch, LogStretch, HistEqStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from scipy.stats import iqr


def display(arr, ax=None, lim=1, mid=0, title=None, fs=16, norm=None, cmap="binary_r", cbar=True, axis=True):
	show = False
	if ax is None:
		fig, ax = plt.subplots(figsize=(10,10))
		show = True

	norm_kw = {}
	if norm == "Centered":
		norm_kw.update({"norm": colors.CenteredNorm(vcenter=mid)})
	elif norm == "Log":
		norm_kw.update({"norm": colors.LogNorm()})
	elif norm == "HistEqStretch":
		arr -= np.median(arr)
		noise = iqr(arr, rng = (16, 84))/2
		norm_kw.update({"norm": ImageNormalize(stretch=HistEqStretch(arr[arr <= 3*noise]), clip=False, vmax=3*noise, vmin=np.min(arr))})
		# norm_kw.update({"norm": ImageNormalize(stretch=HistEqStretch(arr), clip=False)})
	elif norm == "LogStretch":
		arr -= np.median(arr)
		noise = iqr(arr, rng = (16, 84))/2
		arr = np.ma.masked_where(arr < 3*noise, arr)
		norm_kw.update({"norm": ImageNormalize(stretch=LogStretch(), clip=False),
					   "clim": [3*noise, None],
					   "interpolation": "none"})
		
	im = ax.imshow(arr, origin='lower', extent=(-lim, lim, -lim, lim),
				   cmap=cmap, **norm_kw)

	if cbar:
		div = make_axes_locatable(ax)
		cax = div.append_axes("right", size="5%", pad=0.1)
		plt.colorbar(im, cax=cax)
		
	tx = None
	if title is not None:
		tx = ax.set_title(title, fontsize=fs)

	if not axis:
		ax.axis("off")
		
	if show: plt.show()

	return im, tx


