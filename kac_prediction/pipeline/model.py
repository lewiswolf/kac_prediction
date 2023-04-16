'''
'''

# core
from abc import ABC, abstractmethod
from functools import cached_property

# dependencies
import torch

# src
from ..pipeline.types import Parameters

__all__ = ['Model']


class Model(ABC, torch.nn.Module):
	'''
	'''

	@abstractmethod
	class ModelHyperParameters(Parameters):
		'''
		'''
		pass

	def __init__(self) -> None:
		super().__init__()

	@abstractmethod
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		pass

	@abstractmethod
	def loss(
		self,
		y: torch.Tensor,
		y_hat: torch.Tensor,
		carry: dict[str, float | int],
	) -> tuple[torch.nn.Module, dict[str, float | int]]:
		pass

	@abstractmethod
	@cached_property
	def optimiser(self) -> torch.optim.Optimizer:
		pass

	# @abstractmethod
	# def train_loop(self) -> None:
	# 	pass
