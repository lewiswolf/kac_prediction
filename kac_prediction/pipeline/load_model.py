'''
Methods for loading a model and its parameters.
'''

# core
import os
import shlex
import subprocess
from typing import Optional

# dependencies
import torch

# src
from .types import ExportedModel

__all__ = ['loadModel']


def loadModel(Model: type[torch.nn.Module], path_to_parameters: str, url: Optional[str] = None) -> torch.nn.Module:
	'''
	This method checks first if the model parameters exist at the specified path, if not they are downloaded from the
	specified url. Finally, a model is loaded and initialised.
	'''

	# assert params exist
	if not os.path.exists(path_to_parameters):
		assert url is not None, 'Model parameters could not be located, either locally or using a specified url.'
		# assert directory exists
		directory = os.path.dirname(path_to_parameters)
		if not os.path.isdir(directory):
			os.makedirs(directory)
		# download parameters using curl
		# could change this to `pip install wget` if this doesn't work on all os
		subprocess.run(shlex.split(f'curl {url} -L --output {path_to_parameters}'))
	# load parameters
	params: ExportedModel = torch.load(path_to_parameters, map_location='cpu')
	# load model
	m = Model(*params['model_args'].values(), **params['model_kwargs'])
	m.load_state_dict(params['model_state_dict'])
	return m.eval()
