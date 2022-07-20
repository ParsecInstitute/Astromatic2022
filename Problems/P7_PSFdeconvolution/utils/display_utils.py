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
import pandas as pd


def display(arr, ax=None, lim=1, mid=0, title=None, fs=16, norm=None, cmap="binary_r", cbar=True, axis=True, style="default"):
	plt.style.use(style)
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


def display_x_y(x, y, dataset_name=None, size=5, norm=None, style="default"):
	plt.style.use(style)

	size = min(size, x.shape[0])
	fig, ax = plt.subplots(figsize=(10, size * 3), nrows=size, ncols=3)
	if dataset_name is not None:
		fig.suptitle(f"sample from {dataset_name}", fontsize=18, y=1.0)
	for i in range(size):
		if i == 0:
			titles = ["corrupt", "psf", "truth"]
		else:
			titles = [None] * 3

		display(x[i, 0], ax=ax[i, 0], title=titles[0], axis=False, norm=norm, style=style)
		display(x[i, 1], ax=ax[i, 1], title=titles[1], axis=False, norm=norm, style=style)
		display(y[i, 0], ax=ax[i, 2], title=titles[2], axis=False, norm=norm, style=style)

	plt.tight_layout()
	plt.show()


def display_progress(x, yhat, y, epoch=None, size=5, norm=None, style="default"):
	plt.style.use(style)

	size = min(size, x.shape[0])
	fig, ax = plt.subplots(figsize=(16, size * 4), nrows=size, ncols=4)
	if epoch is not None:
		fig.suptitle(f"epoch {epoch}", fontsize=18, y=1.0)

	for i in range(size):
		if i == 0:
			titles = ["corrupt", "reconstruction", "truth", "residuals"]
		else:
			titles = [None] * 4

		display(x[i, 0], ax=ax[i, 0], title=titles[0], axis=False, norm=norm, style=style)
		display(yhat[i, 0], ax=ax[i, 1], title=titles[1], axis=False, norm=norm, style=style)
		display(y[i, 0], ax=ax[i, 2], title=titles[2], axis=False, norm=norm, style=style)
		display(yhat[i, 0] - y[i, 0], ax=ax[i, 3], title=titles[3], axis=False, norm="Centered", cmap="seismic", style=style)

	plt.tight_layout()
	plt.show()


def loss_curves(path_out, loss, model_name=None, save=True, style="default"):
	plt.style.use(style)

	# open train logs
	with open(glob(f"{path_out}/logs/*logs.csv")[0]) as log_file:
		logs_df = pd.read_csv(log_file)

	nrows = 2
	height_ratios = [3.5, 1.4]

	# Train vs Val loss
	fig = plt.figure(figsize=(9, nrows*2), constrained_layout=True)
	gs = gridspec.GridSpec(nrows, 1, figure=fig, height_ratios=height_ratios, hspace=0)

	ax1 = fig.add_subplot(gs[1])
	ax1.semilogy(logs_df["epoch"], logs_df["lr"], "-g", label=r"lr")
	ax1.set_ylabel("lr")
	ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
	ax1.grid(True)

	lns1, lbs1 = ax1.get_legend_handles_labels()
	# ax1.tick_params(axis="y", labelcolor="g")

	ax1.legend(lns1, lbs1, loc="best", fontsize=9)
	ax1.set_xlabel("Epoch")
	ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

	ax0 = fig.add_subplot(gs[0], sharex=ax1)
	ax0.semilogy(logs_df["epoch"], logs_df["valid_loss"], '--r', label=r'validation')
	ax0.semilogy(logs_df["epoch"], logs_df["train_loss"], '-k', label=r'training')

	ax0.set_ylabel(f"{loss}")
	ax0.legend(loc="best", fontsize=9)
	ax0.xaxis.set_major_locator(MaxNLocator(integer=True))
	ax0.tick_params(labelbottom=False)
	ax0.grid(True)

	plt.rcParams['axes.facecolor'] = 'white'
	plt.rcParams['savefig.facecolor'] = 'white'
	if save:
		plt.savefig(path_out + f'/{model_name}_loss_curves.png', dpi=200, bbox_inches="tight")
	plt.show()