'''
Methods for loading a model and its parameters.
'''

# core
# import os
import shlex
import subprocess
from typing import Literal

# dependencies
# import torch

# src
from kac_drumset import (
	# methods
	generateDataset,
	loadDataset,
	transformDataset,
	# types
	AudioSampler,
	RepresentationSettings,
	SamplerSettings,
	TorchDataset,
)
# from .types import Datasets, ExportedModel
from .types import Datasets

__all__ = [
	'importDataset',
	# 'loadModel',
]


def importDataset(
	dataset_dir: str,
	dataset_name: Datasets | Literal[''] = '',
	LocalSampler: type[AudioSampler] | None = None,
	representation_settings: RepresentationSettings = {},
	sampler_settings: SamplerSettings = {'duration': 1., 'sample_rate': 48000},
) -> TorchDataset:
	'''
	Load, download or locally generate a dataset. If a dataset already exists in dataset_dir, then that dataset is loaded
	and transformed if necessary. If the project is run in evaluation mode, the official dataset is downloaded using the
	zenodo script in /bin. Else a small local dataset is generated for testing.
	'''

	# load a dataset normally
	try:
		dataset = transformDataset(loadDataset(dataset_dir=dataset_dir), representation_settings)
	except Exception as e:
		# if a metadata.json does not exist...
		if type(e).__name__ == 'FileNotFoundError':
			assert dataset_name != '' and LocalSampler is not None, \
				'importDataset requires at least a dataset_name or a LocalSampler to generate a new dataset'
			# import the official dataset for this project
			if dataset_name != '':
				subprocess.run(shlex.split(f'sh ./bin/install-dataset.sh {dataset_name}'))
				dataset = transformDataset(loadDataset(dataset_dir=dataset_dir), representation_settings)
			# generate a dataset locally
			else:
				dataset = generateDataset(
					LocalSampler,
					dataset_dir=dataset_dir,
					dataset_size=200,
					representation_settings=representation_settings,
					sampler_settings=LocalSampler.Settings(sampler_settings),
				)
		else:
			raise e
	return dataset


# def loadModel(Model: type[torch.nn.Module], path_to_parameters: str, url: str | None = None) -> torch.nn.Module:
# 	'''
# 	This method checks first if the model parameters exist at the specified path, if not they are downloaded from the
# 	specified url. Finally, a model is loaded and initialised.
# 	'''

# 	# assert params exist
# 	if not os.path.exists(path_to_parameters):
# 		assert url is not None, 'Model parameters could not be located, either locally or using a specified url.'
# 		# assert directory exists
# 		directory = os.path.dirname(path_to_parameters)
# 		if not os.path.isdir(directory):
# 			os.makedirs(directory)
# 		# download parameters using curl
# 		# could change this to `pip install wget` if this doesn't work on all os
# 		subprocess.run(shlex.split(f'curl {url} -L --output {path_to_parameters}'))
# 	# load parameters
# 	params: ExportedModel = torch.load(path_to_parameters, map_location='cpu')
# 	# load model
# 	m = Model(*params['model_args'].values(), **params['model_kwargs'])
# 	m.load_state_dict(params['model_state_dict'])
# 	return m.eval()
