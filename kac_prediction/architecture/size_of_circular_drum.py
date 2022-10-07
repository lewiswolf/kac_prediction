'''
This file contains a neural network for determining the size of a circular drum. The model architecture is based on the
CRePE model for fundamental frequency detection.
'''

# core
from typing import Literal

# dependencies
import torch						# pytorch

__all__ = ['SizeOfCircularDrum']


class SizeOfCircularDrum(torch.nn.Module):
	'''
	A remake of CRePE, a deep CNN for pitch detection, here used for estimation of the size of a drum.
	Source: https://github.com/marl/crepe
	DOI: https://doi.org/10.48550/arXiv.1802.06182
	'''

	class ConvLayer(torch.nn.Module):
		'''
		A single convolutional layer.
		'''

		def __init__(self, filters: int, width: int, stride: int, in_channels: int, dropout: float) -> None:
			'''
			Init convolutional layer.
			params:
				filters: number of convolutional filters.
				width: kernel_size for the convolution.
				stride: stride for the convolution.
				in_channels: input size.
				dropout: probability for dropout nodes.
			'''

			super().__init__()
			p1 = (width - 1) // 2
			p2 = (width - 1) - p1
			self.pad = torch.nn.ZeroPad2d((0, 0, p1, p2))
			self.conv2d = torch.nn.Conv2d(
				in_channels=in_channels,
				out_channels=filters,
				kernel_size=(width, 1),
				stride=(stride, 1),
			)
			self.relu = torch.nn.ReLU()
			self.bn = torch.nn.BatchNorm2d(filters)
			self.pool = torch.nn.MaxPool2d(kernel_size=(2, 1))
			self.dropout = torch.nn.Dropout(dropout)

		def forward(self, x: torch.Tensor) -> torch.Tensor:
			''' Forward pass. '''
			return self.dropout(self.pool(self.bn(self.relu(self.conv2d(self.pad(x))))))

	def __init__(self, depth: Literal['large', 'medium', 'small', 'tiny'], dropout: float) -> None:
		'''
		Initialise CRePE model.
		params:
			depth: limit the size of CRePE for a trade off in accuracy.
			dropout: hyperparameter
		'''

		# init
		super().__init__()
		capacity_multiplier: int = {'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32}[depth]
		# create 6 convolutional layers
		filters: list[int] = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
		widths: list[int] = [512, 64, 64, 64, 64, 64]
		strides: list[int] = [4, 1, 1, 1, 1, 1]
		self.conv_layers = torch.nn.ModuleList(
			[self.ConvLayer(
				filters[n],
				widths[n],
				strides[n],
				1 if n == 0 else filters[n - 1],
				dropout,
			) for n in range(6)],
		)
		# fully connected layer
		self.linear = torch.nn.Linear(64 * capacity_multiplier, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		''' Forward pass. '''

		x = x[:, :1024].view(x.shape[0], 1, -1, 1)
		for layer in self.conv_layers:
			x = layer(x)
		x = x.permute(0, 3, 2, 1)
		x = x.reshape(x.shape[0], -1)
		x = self.linear(x)
		return x
