'''
Utility types used throughout this code base.
'''

# core
from typing import Any, TypedDict, Union

__all__ = [
	'ExportedModel',
	'RunInfo',
]


# Percentage splits that sum to 1.
# DatasetSplit: TypeAlias = Annotated[tuple[float, float, float], lambda x: sum(x) == 1.]
# A probability value ∈ [0., 1.]
# Probability: TypeAlias = Annotated[float, lambda x: x >= 0. and x <= 1.]

class ExportedModel(TypedDict, total=True):
	''' All the info needed to save and load a model. '''
	epoch: int								# current epoch
	evaluation_loss: Union[float, None]		# current evaluation loss, if not a test model
	model_state_dict: dict[str, Any]		# model parameters
	model_args: dict[str, Any]				# model args
	model_kwargs: dict[str, Any]			# model kwargs
	optimizer_state_dict: dict[str, Any]	# current optimiser state
	testing_loss: Union[float, None]		# current testing loss if a test model
	training_loss: float					# current training loss


class RunInfo(TypedDict, total=True):
	''' Info about the current training run. '''
	id: str									# this training session's ID
	model_dir: str							# where should the model be saved locally
