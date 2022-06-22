import sys
import os
sys.path.insert(0, os.path.join(os.getenv("ASTROMATIC_PATH"), "Problems"))
import json
from glob import glob
import argparse
import copy
import time
import numpy as np
import pandas as pd

import torch
from torch.optim import Adam, Adamax
from torch.utils.data import DataLoader
import torch.nn as nn
from torchinfo import summary

from utils.display_utils import loss_curves, display_x_y, display_progress
from torch_datasets import PSFDataset
from vanilla_cnn_solution import VanillaCNN

# plt.style.use("science")

def save_logs(logs_dic, path_out, iter, filename):
	logs_df = pd.DataFrame(data=logs_dic, index=[0])

	if iter == 1:
		wmode = "w"
		header = True
	else:
		wmode = "a"
		header = False

	with open(os.path.join(path_out, "logs", filename), mode=wmode) as csv_file:
		logs_df.to_csv(csv_file, header=header)


def train(model,
		  dataset_dir,
		  data_size,
		  loss_fn,
		  optimizer,
		  lr,
		  model_name,
		  save_model_every,
		  plot,
		  plot_every,
		  style,
		  exp_path,
		  batch_size,
		  epochs,
		  device=torch.device("cpu")):

	criterion = loss_fn().to(device, non_blocking=True)
	optimizer = optimizer(model.parameters(), lr=lr)

	start = time.time()

	dataset = PSFDataset(dir_path=dataset_dir, size=data_size, val_frac=0.2, test_frac=0.2)
	display_norm = None

	epoch_header = True
	step = 0

	for epoch in range(1, epochs + 1):

		dataset.method = "train"
		train_dataloader = DataLoader(copy.copy(dataset), batch_size=batch_size, shuffle=True)

		dataset.method = "val"
		val_dataloader = DataLoader(copy.copy(dataset), batch_size=batch_size, shuffle=True)

		print('\n')
		print(f'Epoch {epoch}/{epochs}')
		print('-' * 10)

		epoch_logs = {"epoch": epoch}

		for phase in ["valid", "train"]:

			if phase == "train":
				model.train(True)
				dataloader = train_dataloader
			else:
				model.train(False)
				dataloader = val_dataloader

			running_loss = 0.

			for i, (x, y) in enumerate(dataloader):
				x = x.to(device, non_blocking=True)
				y = y.to(device, non_blocking=True)

				# training phase
				if phase == "train":
					# display what the data looks like
					if i == 0 and epoch == 1 and plot:
						dset_name = os.path.basename(dataset_dir)
						display_x_y(x.cpu().detach().numpy(), y.cpu().detach().numpy(), dset_name, size=5,
									norm=display_norm, style=style)

					model.zero_grad()

					yhat = model(x)  # forward pass
					loss = criterion(yhat, y)  # evaluate loss
					loss.backward()  # backward pass

					optimizer.step()  # update network

					step += 1

					if plot and epoch % plot_every == 0 and i == 0:
						display_progress(x.cpu().detach().numpy(), yhat.cpu().detach().numpy(),
										 y.cpu().detach().numpy(), epoch, norm=display_norm, style=style)

				# validation phase
				else:
					with torch.no_grad():
						yhat = model(x)
						loss = criterion(yhat, y)

				# update cumulative values
				running_loss += float(loss.detach())

			# after all batches processed
			epoch_loss = running_loss / dataloader.__len__()

			# epoch_logs = {"epoch": epoch,
			# f"{phase}_loss": epoch_loss}

			epoch_logs.update({f"{phase}_loss": epoch_loss})

			if phase == "train":
				epoch_logs.update({"lr": optimizer.param_groups[0]['lr']})

			# display progress
			print(f'epoch {epoch} ==>  {phase} loss: {epoch_loss:.4e}')

		save_logs(epoch_logs, exp_path, epoch, f"logs.csv")

		# Keeping track of the model
		if save_model_every is not None:
			if epoch % save_model_every == 0:
				torch.save(model.state_dict(), exp_path + f'/models/{model_name}_epoch_{epoch:03d}.pt')

	# print training time
	time_elapsed = time.time() - start
	print(f'Training completed in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')

	# **Save model**
	torch.save(model.state_dict(), exp_path + "/models/" + f"{model_name}_complete.pt")

	return


if __name__ == "__main__":

	# --- Training ---------------------------------------------------------
	parser = argparse.ArgumentParser(
		description="Train a neural network to deconvolve corrupted astronomical images")

	parser.add_argument("--path_in", type=str, default=os.path.join(os.getenv("ASTROMATIC_PATH")),
						help="principal repo path")
	parser.add_argument("--path_out", type=str,
						default=os.path.join(os.getenv("ASTROMATIC_PATH"), "Problems", "P7_PSFdeconvolution"),
						help="path to problem dir")
	parser.add_argument("--exp_name", type=str, required=True, help="dir for experiment relative to path_out/experiments/")
	parser.add_argument("--dataset", type=str, default="debug_deconv_dataset", help="name of dataset directory")
	parser.add_argument("--npix", type=int, required=True, help="pixel size of images in dataset")
	parser.add_argument("--data_size", type=int, default=None, help="size of subset of dataset")
	parser.add_argument("--batch_size", type=int, default=4, help="batch size, used if not in HP file")
	parser.add_argument("--n_epochs", type=int, default=3, help="number of training epochs")
	parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
	parser.add_argument("--loss", type=str, default="MSE", help="type of loss function")
	parser.add_argument("--optimizer", type=str, default="Adamax", help="type of optimizer")
	parser.add_argument("--nla", type=str, default="ReLU", help="type of non-linear activation")
	parser.add_argument("--model_type", type=str, default="VanillaCNN", help="specifies which type of model to train")
	parser.add_argument("--model_name", type=str, default="model", help="name under which to save the model")
	parser.add_argument("--save_model_every", type=int, default=1, help="save model at every n epochs")
	parser.add_argument("--plot", action="store_true")
	parser.add_argument("--plot_every", type=int, default=50, help="epoch interval at which to plot progress")
	parser.add_argument("--plt_style", type=str, default="default", help="pyplot style")
	parser.add_argument("--seed", type=int, default=None, help="random seed")

	args = parser.parse_args()

	if args.seed is not None:
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)

	exp_path = os.path.join(args.path_out, "experiments", args.exp_name)

	if not os.path.exists(os.path.join(exp_path, "models")):
		os.makedirs(os.path.join(exp_path, "models"))
	if not os.path.exists(os.path.join(exp_path, "logs")):
		os.makedirs(os.path.join(exp_path, "logs"))

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# - loss function -
	if args.loss == "MSE":
		loss_fct = nn.MSELoss

	# - optimizer -
	if args.optimizer == "Adamax":
		optimizer = Adamax

	in_ch = 2
	out_ch = 1

	# --- Model ---
	if args.model_type == "VanillaCNN":
		model = VanillaCNN(npix=args.npix, in_ch=in_ch, out_ch=out_ch, activation=args.nla).float()
	else:
		raise ValueError(f"model of type {args.model_type} not found")

	summary(model, input_size=(args.batch_size, in_ch, args.npix, args.npix))

	model = model.to(device, non_blocking=True)

	train(model=model,
			   dataset_dir=os.path.join(args.path_out, "datasets", args.dataset),
			   data_size=args.data_size,
			   loss_fn=loss_fct,
			   optimizer=optimizer,
			   lr=args.lr,
			   model_name=args.model_name,
			   save_model_every=args.save_model_every,
			   plot=args.plot,
		  	   plot_every=args.plot_every,
			   style=args.plt_style,
			   exp_path=exp_path,
			   batch_size=args.batch_size,
			   epochs=args.n_epochs,
			   device=device)

	# --- Results ----------------------------------------------------------
	if args.plot:
		loss_curves(exp_path, args.loss, model_name=args.model_name, save=False, style=args.plt_style)
