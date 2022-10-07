'''
This file is used to prepare a dataset for multiple purposes. Given a three-way split,
this file will return three subsets prepared for either training, testing or evaluating
a neural network.
'''

# core
from itertools import accumulate

# dependencies
import torch		# pytorch

# src
from kac_drumset import TorchDataset

__all__ = [
	'inferSubdivisions',
	'getEvaluationDataset',
	'getTestingDataset',
	'getTrainingDataset',
]


def inferSubdivisions(dataset_size: int, split: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> list[int]:
	'''
	Calculates the integer splits for the training, testing and validation sets.
	'''

	subdivisions = [round(dataset_size * p) for p in split]
	# correct errors
	# this correction supposes that split[0] > split[1 or 2]
	subdivisions[0] += dataset_size - sum(subdivisions)
	# cumulative sums
	return list(accumulate(subdivisions))


def getEvaluationDataset(
	dataset: TorchDataset,
	split: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> torch.utils.data.DataLoader:
	'''
	Extracts the evaluation subset from the dataset, and returns a DataLoader ready
	for evaluating a network.
	'''

	subdivisions = inferSubdivisions(dataset.__len__(), split)
	return torch.utils.data.DataLoader(
		dataset,
		sampler=torch.utils.data.SubsetRandomSampler(list(range(subdivisions[1], subdivisions[2]))),
	)


def getTestingDataset(
	dataset: TorchDataset,
	split: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> torch.utils.data.DataLoader:
	'''
	Extracts the training subset from the dataset, and returns a DataLoader ready
	for testing a network.
	'''

	subdivisions = inferSubdivisions(dataset.__len__(), split)
	return torch.utils.data.DataLoader(
		dataset,
		sampler=torch.utils.data.SubsetRandomSampler(list(range(subdivisions[0], subdivisions[1]))),
	)


def getTrainingDataset(
	dataset: TorchDataset,
	batch_size: int,
	split: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> torch.utils.data.DataLoader:
	'''
	Extracts the training subset from the dataset, and returns a DataLoader ready
	for training a network.
	'''

	subdivisions = inferSubdivisions(dataset.__len__(), split)
	return torch.utils.data.DataLoader(
		dataset,
		batch_size=batch_size,
		sampler=torch.utils.data.SubsetRandomSampler(list(range(0, subdivisions[0]))),
	)
