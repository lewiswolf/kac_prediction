'''
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
	epoch: int
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
	def innerTrainingLoop(self, loop_length: int, x: torch.Tensor, y: torch.Tensor) -> None:
		'''
		This inner training loop should be designed to satisfy the loop:
			for (x, y) in training_dataset:
				Model.innerTrainingLoop(len(training_dataset), x.to(device), y.to(device))
		and should somewhere include the line:
			self.training_loss += ...
		'''
		pass
