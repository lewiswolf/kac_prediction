'''
Utility types used throughout this code base.
'''

# core
from typing import TypedDict

__all__ = [
	'RunInfo',
]


# Percentage splits that sum to 1.
# DatasetSplit: TypeAlias = Annotated[tuple[float, float, float], lambda x: sum(x) == 1.]
# A probability value ∈ [0., 1.]
# Probability: TypeAlias = Annotated[float, lambda x: x >= 0. and x <= 1.]


class RunInfo(TypedDict):
	''' Info about the current training run. '''
	id: str
	model_dir: str
