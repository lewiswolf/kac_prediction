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

__all__ = ['loadModel']


def loadModel(Model: type[torch.nn.Module], path_to_parameters: str, url: Optional[str] = None) -> torch.nn.Module:
	''' Load and initialise a model. '''

	# download parameters using curl
	# could change this to `pip install wget` if this doesn't work on all os
	if not os.path.exists(path_to_parameters):
		assert url is not None, 'Model parameters could not be located, either locally or using a specified url.'
		subprocess.run(shlex.split(f'curl {url} -L --output {path_to_parameters}'))
	# load parameters
	params = torch.load(path_to_parameters, map_location='cpu')
	# load model
	m = Model(*params['model_args'].values(), **params['model_kwargs'])
	m.load_state_dict(params['model_state_dict'])
	return m
