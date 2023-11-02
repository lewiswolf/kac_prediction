'''
Methods for loading a model and its parameters.
'''

# core
import os
import shlex
import subprocess

# dependencies
import torch

# src
from .types import ExportedModel, Parameters

__all__ = ['loadModel']


def loadModel(Model: type[torch.nn.Module], path_to_parameters: str, url: str | None = None) -> torch.nn.Module:
	'''
	This method can load a model using parameters derived from multiple locations. Firstly, if you have supplied a
	directory of .pt files, the method will load the parameters with the lowest testing_loss. If you have provided a path
	directly to a specific .pt, the method will load that file instead. And finally, if your supplied path is not a
	directory, but is a .pt file that does not yet exist, you can provide the optional argument url to download these
	parameters for your model and store them at this location.
	'''
	# if it is a directory, check to find the lowest testing_loss
	if os.path.isdir(path_to_parameters):
		if len(os.listdir(path_to_parameters)) != 0:
			testing_loss = 10
			optimum_path = 'parameters.pt'
			for path in os.listdir(path_to_parameters):
				params_tmp = torch.load(os.path.join(path_to_parameters, path), map_location='cpu')
				loss = params_tmp['evaluation_loss']['aggregate'] or params_tmp['testing_loss']['aggregate']
				if testing_loss > loss:
					testing_loss = loss
					optimum_path = path
			path_to_parameters = os.path.join(path_to_parameters, optimum_path)
		else:
			# return a default case if directory is empty
			path_to_parameters = os.path.join(path_to_parameters, 'parameters.pt')
	# assert params exist
	assert os.path.splitext(path_to_parameters)[1] == '.pt', \
		'Your provided path is either a directory that does not exist, or you are not prescribing the location of a .pt file.'
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
	assert params['run_info']['model'] is not None and params['run_info']['model']['name'] == Model.__name__, \
		'The parameters you have loaded do not match the model you have provided.'
	# load model
	m = Model(
		**{key: value for key, value in params['hyperparameters'].items() if key not in Parameters.__annotations__.keys()},
	)
	m.load_state_dict(params['model_state_dict'])
	return m.eval()
