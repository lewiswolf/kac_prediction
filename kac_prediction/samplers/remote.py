from typing import Literal, TypeAlias

__all__ = ['Datasets']

# string literals for installable datasets
Datasets: TypeAlias = Literal[
	'2000-convex-polygon-drums-of-varying-area',
	'2000-elliptic-drums-of-varying-area',
	'2000-irregular-star-drums-of-varying-area',
	'2000-regular-star-drums-of-varying-area',
	'2000-simple-polygon-drums-of-varying-area',
	'5000-circular-drums-of-varying-size',
	'5000-rectangular-drums-of-varying-dimension',
]
