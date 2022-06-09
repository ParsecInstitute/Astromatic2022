import torch
import torch.nn as nn

def select_activation(activ_str, activation_kw={}, inplace=False):
	if activ_str.lower() == "relu":
		activation = nn.ReLU(inplace=inplace)
	elif activ_str.lower() == "elu":
		activation = nn.ELU(inplace=inplace, **activation_kw)
	elif activ_str.lower() == "leakyrelu":
		activation = nn.LeakyReLU(inplace=inplace, **activation_kw)
	elif activ_str.lower() == "tanh":
		activation = nn.Tanh()
	else:
		raise ValueError(f"Unrecognized activation type '{activ_str}'")

	return activation


class VanillaCNN(nn.Module):
	def __init__(self, in_ch=2, activation="ReLU"):
		"""
		A simple CNN implementation for image-to-image regression. Works for npix=129
		:param in_ch:
		:param activation:
		"""
		super().__init__()

		self.activation = select_activation(activ_str=activation)

		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels=in_ch, out_channels=16, kernel_size=4, stride=4, padding=1), 	# in: (2,129,129) | out: (16,32,32)
			self.activation(),
			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding="same"), 			# in: (16,32,32) | out: (32,32,32)
			self.activation(),
			nn.AvgPool2d(kernel_size=2), 														# in: (32,32,32) | out: (32,16,16)
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same"), 			# in: (32,16,16) | out: (64,16,16)
			self.activation(),
			nn.AvgPool2d(kernel_size=2),  														# in: (64,16,16) | out: (64,8,8)
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same"), 		# in: (64,8,8) | out: (128,8,8)
			self.activation(),
			nn.AvgPool2d(kernel_size=2), 														# in: (128,8,8) | out: (128,4,4)
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding="same"),  		# in: (128,4,4) | out: (256,4,4)
			self.activation()
		)

		self.decoder = nn.Sequential(
			nn.UpsamplingBilinear2d(scale_factor=2),											# in: (256,4,4) | out: (256,8,8)
			nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding="same"),		# in: (256,8,8) | out: (128,8,8)
			self.activation(),
			nn.UpsamplingBilinear2d(scale_factor=2),  											# in: (128,8,8) | out: (128,16,16)
			nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding="same"),			# in: (128,16,16) | out: (64,16,16)
			self.activation(),
			nn.UpsamplingBilinear2d(scale_factor=2),  											# in: (64,16,16) | out: (64,32,32)
			nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding="same"),			# in: (64,32,32) | out: (32,32,32)
			self.activation(),
			nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding="same"),			# in: (32,32,32) | out: (16,32,32)
			self.activation(),
			nn.ConvTranspose2d()	# TODO
		)