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
	def __init__(self, npix, in_ch=2, out_ch=1, activation="ReLU"):
		"""
		A simple CNN implementation for image-to-image regression. Works for npix=129
		:param in_ch:
		:param activation:
		"""
		super().__init__()

		self.activation = select_activation(activ_str=activation, inplace=True)

		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels=in_ch, out_channels=16, kernel_size=3, padding="same"), 		# in: (2,npix,npix) | out: (16,npix,npix)
			self.activation,
			nn.AvgPool2d(kernel_size=2),														# in (16, npix, npix) | out: (16, npix//2, npix//2
			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding="same"), 			# in: (16,npix//2,npix//2) | out: (32,npix//2,npix//2)
			self.activation,
			nn.AvgPool2d(kernel_size=2), 														# in: (32,npix//2,npix//2) | out: (32,npix//4,npix//4)
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same"), 			# in: (32,npix//4,npix//4) | out: (64,npix//4,npix//4)
			self.activation,
			nn.AvgPool2d(kernel_size=2),  														# in: (64,npix//4,npix//4) | out: (64,npix//8,npix//8)
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same"), 		# in: (64,npix//8,npix//8) | out: (128,npix//8,npix//8)
			self.activation,
			nn.AvgPool2d(kernel_size=2), 														# in: (128,npix//8,npix//8) | out: (128,npix//16,npix//16)
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding="same"),  		# in: (128,npix//16,npix//16) | out: (256,npix//16,npix//16)
			self.activation
		)

		self.decoder = nn.Sequential(
			nn.UpsamplingBilinear2d(scale_factor=2),											# in: (256,npix//16,npix//16) | out: (256,npix//8,npix//8)
			nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding="same"),		# in: (256,npix//,npix//8) | out: (128,npix//8,npix//8)
			self.activation,
			nn.UpsamplingBilinear2d(scale_factor=2),  											# in: (128,npix//8,npix//8) | out: (128,npix//4,npix//4)
			nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding="same"),			# in: (128,npix//4,npix//4) | out: (64,npix//4,npix//4)
			self.activation,
			nn.UpsamplingBilinear2d(scale_factor=2),  											# in: (64,npix//4,npix//4) | out: (64,npix//2,npix//2)
			nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding="same"),			# in: (64,npix//2,npix//2) | out: (32,npix//2,npix//2)
			self.activation,
			nn.UpsamplingBilinear2d(scale_factor=2),											# in: (32,npix//2,npix//2) | out: (32,npix,npix)
			nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding="same"),			# in: (32,npix,npix) | out: (16,npix,npix)
			self.activation,
			nn.Conv2d(in_channels=16, out_channels=out_ch, kernel_size=3, padding="same")			# in: (16,npix,npix) | out: (1,npix,npix)
		)

	def forward(self, x):
		x = self.encoder(x)
		return self.decoder(x)


