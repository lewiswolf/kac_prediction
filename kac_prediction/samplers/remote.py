from typing import Literal, TypeAlias

__all__ = ['Datasets']

# string literals for installable datasets
Datasets: TypeAlias = Literal[
	'2000-convex-polygonal-drums-of-varying-size',
	'5000-circular-drums-of-varying-size',
	'5000-rectangular-drums-of-varying-dimension',
]
