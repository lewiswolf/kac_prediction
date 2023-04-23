'''
Utility types used throughout this code base.
'''

# core
from typing import Any, Literal, TypeAlias, TypedDict

__all__ = [
	'Datasets',
	'ExportedModel',
	'Parameters',
	'RunInfo',
]


# string literals for installing datasets
Datasets: TypeAlias = Literal[
	'2000-convex-polygonal-drums-of-varying-size',
	'5000-circular-drums-of-varying-size',
	'5000-rectangular-drums-of-varying-dimension',
]


class ModelInfo(TypedDict, total=True):
	'''
	Information about the model used during a training loop.
	'''
	name: str		# name of the model
	version: str	# version of kac_prediction the model originates from


class Parameters(TypedDict, total=True):
	''' Default parameters for any model. '''
	batch_size: int								# batch size
	dataset_split: tuple[float, float, float]	# how is the dataset split between training, testing and evaluation
	num_of_epochs: int							# number of epochs
	testing: bool								# is the network being used for testing
	with_early_stopping: bool					# should the network stop if it has reached a minima


class RunInfo(TypedDict, total=True):
	''' Info about the current training run. '''
	epoch: int
	exports_dir: str							# absolute path to where the model should be saved locally
	id: str										# this training session's ID
	model: ModelInfo | None


class ExportedModel(TypedDict, total=True):
	''' All the info needed to save and load a model. '''

	dataset: dict[str, Any]						# metadata imported from TorchDataset
	hyperparameters: dict[str, Any]				# a copy of ModelHyperParameters
	evaluation_loss: dict[str, Any] | None		# current evaluation loss, if not a test model
	model_state_dict: dict[str, Any]			# model parameters
	optimizer_state_dict: dict[str, Any]		# current optimiser state
	run_info: RunInfo							# a copy of RunInfo
	testing_loss: dict[str, Any] | None			# current testing loss if a test model
	training_loss: float						# current training loss
