'''
This file contains the loadDataset method.
'''

# core
import json
import math
import os

# dependencies
from tqdm import tqdm			# CLI progress bar
import torch					# pytorch

# src
from .dataset import TorchDataset
from .input_representation import InputRepresentation
from .utils import listToTensor
from ..utils import printEmojis, tqdm_format

__all__ = [
	'loadDataset',
]


def loadDataset(dataset_dir: str) -> TorchDataset:
	'''
	loadDataset imports a kac_drumset dataset from the directory specified by the absolute path dataset_dir.
	'''

	try:
		with open(os.path.normpath(f'{dataset_dir}/metadata.json')) as file:
			# import metadata
			file.readlines(1)
			dataset_size = int(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1])
			representation_settings = json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1])
			sampler = json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1])
			sampler_settings = json.loads(file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1])
			file.readlines(1)
			# create dataset
			dataset = TorchDataset(
				dataset_dir=dataset_dir,
				dataset_size=dataset_size,
				representation_settings=representation_settings,
				sampler=sampler,
				sampler_settings=sampler_settings,
				x_size=InputRepresentation.transformShape(
					math.ceil(sampler_settings['duration'] * sampler_settings['sample_rate']),
					representation_settings,
				),
			)
			# import loop
			printEmojis('Importing dataset... 📚')
			with tqdm(total=dataset_size, bar_format=tqdm_format, unit=' data samples') as bar:
				for i in range(dataset_size):
					# import relevant information
					file.readlines(1)
					x = torch.as_tensor(json.loads(
						file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:-1],
					))
					y = listToTensor(json.loads(
						file.readlines(1)[0].replace('\n', '').split(':', 1)[1][1:],
					))
					file.readlines(1)
					# append input features to dataset
					dataset.__setitem__(i, x, y)
					bar.update(1)
			file.close()
	except Exception as e:
		if type(e).__name__ == 'IndexError':
			raise IndexError('The dataset you are importing is corrupted.')
		if type(e).__name__ == 'FileNotFoundError':
			raise FileNotFoundError('The dataset you tried to import does not exist.')
	return dataset
