'''
The idea behind this design is that a Routine can abstract all of the complicated communications between both pytorch
and wandb, whilst simultaneously providing a central namespace for programming inside and outside of the routine.
'''

# core
from importlib.metadata import version
from itertools import accumulate
import os
import random
import shlex
import string
import subprocess
from typing import Any, Callable, Literal
import yaml

# dependencies
import torch			# pytorch
from tqdm import tqdm	# progress bar
import wandb			# experiment tracking

# src
from .model import Model
from .types import Datasets, ExportedModel, Parameters, RunInfo
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
from kac_drumset.utils import clearDirectory, printEmojis


class Routine:
	'''
	This class houses an entire training/model generation schema, and abstracts several key methods for customisability.
	Each routine should be implemented as follows.

	# initialise a default routine
	routine = Routine(
		# directory to store saved models
		exports_dir=...,
		# configuration for wandb
		wandb_config=...,
	)

	# initialise parameters
	routine.setParameters(
		# default parameters
		SomeModel.ModelHyperParameters(*args),
		# yaml config path
		config_path=...,
	)

	# load, generate or install a dataset
	routine.importDataset(*args)

	# shape data
	routine.D.X = ...
	routine.D.Y = ...

	# configure model
	routine.M = SomeModel(*routine.P)

	# define how the model is to be tested
	def innerTestingLoop(i: int, loop_length: float, x: torch.Tensor, y: torch.Tensor) -> None:
		...

	# train and test a model
	routine.train(innerTestingLoop)
	'''

	device: torch.device 		# device
	D: TorchDataset				# dataset
	M: Model					# Model
	P: dict[str, Any]			# hyperparameters
	R: RunInfo					# information about the training run
	using_wandb: bool			# hidden flag for wandb

	def __init__(self, exports_dir: str = '', wandb_config: dict[str, Any] = {}) -> None:
		'''
		The init method initialises self.device, self.using_wandb and self.R.
		params:
			exports_dir		directory for the local model
			wandb_config	initialise wandb
							- {} is a local call without wandb
							- passing wandb_config is a normal call
		'''
		# set device
		if torch.cuda.is_available():
			self.device = torch.device('cuda')
		else:
			printEmojis('😓 WARNING 😓 Nvidia GPU support is not available for training the network.')
			self.device = torch.device('cpu')
		# initialise weights and biases
		self.using_wandb = wandb_config != {}
		if self.using_wandb:
			wandb.init(**wandb_config)
		# create wandb id and exports_dir
		if wandb.run is not None:
			exports_dir = wandb.run.dir
			local_id = wandb.run.id
		# create local id and exports_dir
		else:
			local_id = ''.join(random.choice(string.ascii_letters) for x in range(10))
			exports_dir = f'{exports_dir if exports_dir != "" else "."}/{local_id}'
			os.makedirs(exports_dir)
		# create RunInfo
		self.R = {
			'epoch': 0,
			'exports_dir': exports_dir,
			'id': local_id,
			'model': None,
		}

	def importDataset(
		self,
		dataset_dir: str,
		dataset_name: Datasets | Literal[''] = '',
		LocalSampler: type[AudioSampler] | None = None,
		representation_settings: RepresentationSettings = {},
		sampler_settings: SamplerSettings = {'duration': 1., 'sample_rate': 48000},
	) -> None:
		'''
		Load, download or locally generate a dataset and store it as self.D. If a dataset already exists in dataset_dir, then
		that dataset is loaded and transformed if necessary. If the project is run in evaluation mode, the official dataset
		is downloaded using the zenodo script in /bin. Else a small local dataset is generated for testing.
		'''
		# local dataset error
		class DatasetError(Exception):
			pass
		# load a dataset normally
		try:
			dataset = transformDataset(loadDataset(dataset_dir=dataset_dir), representation_settings)
			if LocalSampler is not None and dataset.sampler['name'] != LocalSampler.__name__:
				raise DatasetError('Default dataset generator does not match the dataset stored in dataset_dir.')
		except Exception as e:
			# if a metadata.json does not exist...
			if type(e).__name__ == 'DatasetError' or type(e).__name__ == 'FileNotFoundError':
				assert dataset_name != '' or LocalSampler is not None, \
					'importDataset() requires at least a dataset_name or a LocalSampler to produce a dataset.'
				# import the official dataset for this project
				if dataset_name != '':
					subprocess.run(shlex.split(f'sh ./bin/install-dataset.sh {dataset_name}'))
					dataset = transformDataset(loadDataset(dataset_dir=dataset_dir), representation_settings)
				# or generate a dataset locally
				if LocalSampler is not None:
					dataset = generateDataset(
						LocalSampler,
						dataset_dir=dataset_dir,
						dataset_size=200,
						representation_settings=representation_settings,
						sampler_settings=sampler_settings,
					)
			else:
				raise e
		self.D = dataset

	def setModel(self, M: Model) -> None:
		''' Set the routine's neural network and update RunInfo. '''
		self.M = M
		self.R['model'] = {'name': type(M).__name__, 'version': version('kac_prediction')}

	def setParameters(self, default: Parameters, config_path: str = '') -> None:
		'''
		This method initialises weights and biases if it is being used, and creates the variable self.P using either a
		default parameter value, a custom yaml file, or by being inferred from weights and biases.
		params:
			config_path		path to custom yaml file parameters
			default			default parameters
		'''
		# init parameters
		self.P = {key: value for key, value in default.items()}
		# load a yaml config file for a single run
		if config_path != '':
			with open(config_path, 'r') as f:
				yaml_file = yaml.load(f.read(), Loader=yaml.FullLoader) or {}
				self.P.update({key: yaml_file[key] if key in yaml_file else value for key, value in self.P.items()})
		# update with wandb.config
		if self.using_wandb:
			self.P.update({key: wandb.config[key] if key in wandb.config else value for key, value in self.P.items()})
			wandb.config.update(self.P)

	def train(self, innerTestingLoop: Callable[[int, int, torch.Tensor, torch.Tensor], None]) -> None:
		'''
		This method runs the entire training and testing schema, beginning by splitting a dataset, running the training and
		testing loops, and exporting/saving the trained model. This method takes as its argument an innerTestingLoop(), which
		should be designed to satisfy the loop:
			for i, (x, y) in enumerate(testing_dataset):
				innerTestingLoop(i, len(testing_dataset), x.to(device), y.to(device))
		and should somewhere include the line:
			self.testing_loss[aggregate] += ...
		'''
		# handle errors
		assert hasattr(self, 'D'), 'Routine.D: TorchDataset is not set.'
		assert hasattr(self, 'M'), 'Routine.M: Model is not set - run Routine.setModel().'
		assert hasattr(self, 'P'), 'Routine.P: Parameters is not set - run Routine.setParameters().'
		# split dataset
		subdivisions = [round(self.D.__len__() * p) for p in self.P['dataset_split']]
		subdivisions[0] += self.D.__len__() - sum(subdivisions) # this correction supposes that split[0] > split[1 or 2]
		subdivisions = list(accumulate(subdivisions))[0:3]
		training_dataset = torch.utils.data.DataLoader(
			self.D,
			batch_size=min(self.P['batch_size'], subdivisions[0]) if self.P['batch_size'] > 0 else subdivisions[0],
			sampler=torch.utils.data.SubsetRandomSampler(list(range(0, subdivisions[0]))),
		)
		testing_dataset = torch.utils.data.DataLoader(
			self.D,
			batch_size=max(min(
				self.P['batch_size'],
				subdivisions[1] - subdivisions[0] if self.P['testing'] else subdivisions[2] - subdivisions[1],
			), 1),
			sampler=torch.utils.data.SubsetRandomSampler(list(range(
				subdivisions[0 if self.P['testing'] else 1],
				subdivisions[1 if self.P['testing'] else 2],
			))),
		)
		# loops
		printEmojis('Training neural network... 🧠')
		if self.using_wandb:
			wandb.watch(self.M, log_freq=1000)
		self.M = self.M.to(self.device)
		early_stopping_cache: tuple[list[float], float] | None = ([], 10.) if self.P['with_early_stopping'] else None
		bar_format: str = '{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}, Elapsed: {elapsed}, ETA: {remaining}, {rate_fmt} '
		with tqdm(bar_format=bar_format, total=self.P['num_of_epochs'], unit='  epochs') as epoch_bar:
			with tqdm(bar_format=bar_format, total=len(training_dataset), unit=' batches') as i_bar:
				with tqdm(bar_format=bar_format, total=len(testing_dataset), unit=' batches') as t_bar:
					while self.P['num_of_epochs'] == 0 or self.R['epoch'] < self.P['num_of_epochs']:
						# initialise
						i_bar.reset()
						t_bar.reset()
						self.M.training_loss = 0.
						self.M.testing_loss['aggregate'] = 0.
						# training
						self.M.train()
						for i, (x, y) in enumerate(training_dataset):
							self.M.innerTrainingLoop(i, len(training_dataset), x.to(self.device), y.to(self.device))
							i_bar.update(1)
						# evaluation / testing
						self.M.eval()
						with torch.no_grad():
							for i, (x, y) in enumerate(testing_dataset):
								innerTestingLoop(i, len(testing_dataset), x.to(self.device), y.to(self.device))
								t_bar.update(1)
						# save model
						try:
							if self.R['epoch'] % 5 == 0 or self.R['epoch'] > 40:
								torch.save(ExportedModel({
									'dataset': {
										"dataset_size": self.D.__len__(),
										"representation_settings": self.D.representation_settings,
										"sampler": self.D.sampler,
										"sampler_settings": self.D.sampler_settings,
									},
									'evaluation_loss': self.M.testing_loss if not self.P['testing'] else None,
									'hyperparameters': self.P,
									'model_state_dict': self.M.state_dict(),
									'optimizer_state_dict': self.M.optimiser.state_dict(),
									'run_info': self.R,
									'testing_loss': self.M.testing_loss if self.P['testing'] else None,
									'training_loss': self.M.training_loss,
								}), f'{self.R["exports_dir"]}/epoch_{self.R["epoch"]}.pt')
								if self.using_wandb:
									wandb.save(
										os.path.normpath(f'{self.R["exports_dir"]}/epoch_{self.R["epoch"]}.pt'),
										self.R["exports_dir"],
									)
						except Exception as e:
							if self.using_wandb:
								wandb.log({'error': e}, step=self.R['epoch'])
							else:
								raise e
						# early stopping
						if early_stopping_cache is not None:
							early_stopping_cache[0].append(self.M.testing_loss['aggregate'])
							early_stopping_cache = (
								early_stopping_cache[0][-32:],
								min(early_stopping_cache[1], self.M.testing_loss['aggregate']),
							)
							if self.R['epoch'] > 32 and (min(early_stopping_cache[0]) > early_stopping_cache[1]):
								break
						# cleanup
						if torch.cuda.is_available():
							torch.cuda.empty_cache()
						# progress bar
						epoch_bar.update(1)
						self.R['epoch'] += 1
		# exit
		if self.P['testing']:
			clearDirectory(self.R['exports_dir'])
