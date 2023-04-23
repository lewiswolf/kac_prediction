'''
This file defines the template class for a neural network, which is just a small wrapper around a torch.nn.Module.
'''

# core
from abc import ABC, abstractmethod

# dependencies
import torch

# src
from ..pipeline.types import Parameters

__all__ = ['Model']


class Model(ABC, torch.nn.Module):
	'''
	Template for a neural network.
	'''

	criterion: torch.nn.Module
	optimiser: torch.optim.Optimizer
	training_loss: float
	testing_loss: float

	@abstractmethod
	class ModelHyperParameters(Parameters):
		''' Template for custom hyper parameters. '''
		pass

	def __init__(self) -> None:
		super().__init__()
		self.training_loss = 0.
		self.testing_loss = 0.

	@abstractmethod
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		''' torch.nn.Module.forward() '''
		pass

	@abstractmethod
	def innerTrainingLoop(self, i: int, loop_length: int, x: torch.Tensor, y: torch.Tensor) -> None:
		'''
		This inner training loop should be designed to satisfy the loop:
			for i, (x, y) in enumerate(training_dataset):
				Model.innerTrainingLoop(i, len(training_dataset), x.to(device), y.to(device))
		and should somewhere include the line:
			self.training_loss += ...
		'''
		pass
