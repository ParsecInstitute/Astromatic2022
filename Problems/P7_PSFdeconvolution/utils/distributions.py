import numpy as np

class Uniform():
	def __init__(self, low=None, high=None, size=None):

		self.low = low
		self.high = high
		self.size = size

	def sample(self):
		return np.random.uniform(low=self.low, high=self.high, size=self.size)


class Poisson():
	def __init__(self, lam=None, size=None):

		self.lam = lam
		self.size = size

	def sample(self):
		return np.random.poisson(self.lam, size=self.size)


class Normal():
	def __init__(self, mu=0., noise_scale=1., size=None):

		self.mu = mu
		self.noise_scale = noise_scale
		self.size = size

	def sample(self, img=1.):

		return np.random.normal(loc=self.mu, scale=self.noise_scale*np.max(np.abs(img)), size=self.size)


class DiracDelta():
	def __init__(self, value):

		self.value = value

	def sample(self):
		return self.value