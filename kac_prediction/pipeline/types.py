# core
from typing import Annotated
from typing_extensions import TypeAlias

# dependencies
from annotated_types import Interval	# support for typing.Annotated

__all__ = [
	'DatasetSplit',
	'Probability',
]


# Percentage splits that sum to 1.
DatasetSplit: TypeAlias = Annotated[tuple[float, float, float], sum(1.)]
# A probability value ∈ [0., 1.]
Probability: TypeAlias = Annotated[float, Interval(ge=0., le=1.)]
