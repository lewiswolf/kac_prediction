'''
'''

# core
import os
import random
import shlex
import string
import subprocess
from typing import Any, Literal
import yaml

# dependencies
import torch			# pytorch
# from tqdm import tqdm	# progress bar
import wandb			# experiment tracking

# src
from .model import Model
from .types import Datasets, Parameters, RunInfo
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
from kac_drumset.utils import printEmojis


class Routine:
	'''
	'''

	device: torch.device 		# device
	D: TorchDataset				# dataset
	M: Model					# Model
	P: dict[str, Any]			# hyperparameters
	R: RunInfo					# information about the training run
	_using_wandb: bool			# hidden flag for wandb

	def __init__(self, model_dir: str = '', wandb_config: dict[str, Any] = {}) -> None:
		'''
		The init method initialises the training device.
		params:
			model_dir		directory for the local model
			wandb_config	initialise wandb
							- None is a local call without wandb
							- passing wandb_config is a normal call
		'''
		# set device
		if torch.cuda.is_available():
			self.device = torch.device('cuda')
		else:
			printEmojis('😓 WARNING 😓 Nvidia GPU support is not available for training the network.')
			self.device = torch.device('cpu')
		# initialise weights and biases
		self._using_wandb = wandb_config != {}
		if self._using_wandb:
			wandb.init(**wandb_config)
		if wandb.run is not None:
			self.R = {'id': wandb.run.id, 'model_dir': wandb.run.dir}
		# create local run and model_dir
		else:
			local_id: str = ''.join(random.choice(string.ascii_letters) for x in range(10))
			model_dir = f'{model_dir if model_dir != "" else "."}/run_{local_id}'
			os.makedirs(model_dir)
			self.R = {'id': local_id, 'model_dir': model_dir}

	def importDataset(
		self,
		dataset_dir: str,
		dataset_name: Datasets | Literal[''] = '',
		LocalSampler: type[AudioSampler] | None = None,
		representation_settings: RepresentationSettings = {},
		sampler_settings: SamplerSettings = {'duration': 1., 'sample_rate': 48000},
	) -> None:
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
		self.D = dataset

	def setParameters(self, default: Parameters, config_path: str = '') -> None:
		'''
		This method initialises weights and biases if it is being used, and creates the variable self.P using either a
		default parameter value, a custom yaml file, or by being inferred from weights and biases.
		params:
			config_path		path to custom yaml file parameters
			default			default parameters
		'''
		# handle errors
		assert hasattr(self, '_using_wandb'), 'getRunInfo must be ran before getParameters'
		# init parameters
		self.P = {key: value for key, value in default.items()}
		# load a yaml config file for a single run
		if config_path != '':
			with open(config_path, 'r') as f:
				yaml_file = yaml.safe_load(f) or {}
				self.P.update({key: yaml_file[key] if key in yaml_file else value for key, value in self.P.items()})
		# update with wandb.config
		if self._using_wandb:
			self.P.update({key: wandb.config[key] if key in wandb.config else value for key, value in self.P.items()})
			wandb.config.update(self.P)

	def train(self) -> None:
		'''
		'''
		assert hasattr(self, 'D'), 'Routine.D: TorchDataset is not set.'
		assert hasattr(self, 'M'), 'Routine.M: Model is not set.'
		assert hasattr(self, 'P'), 'Routine.P: Parameters is not set. Run Routine.getParameters()'
		assert hasattr(self, 'R'), 'Routine.R: RunInfo is not set. Run Routine.getRunInfo()'

		if self._using_wandb:
			wandb.watch(self.M, log_freq=1000)

	# 	# init dataset
	# 	training_dataset, testing_dataset = splitDataset(self.D, self.P['batch_size'], self.P['testing'])
	# 	# init loss
	# 	loss: torch.nn.Module
	# 	training_loss: dict[str, float | int] = {}
	# 	testing_loss: dict[str, float | int] = {}
	# 	test_loss_arr: list[float] = []
	# 	test_loss_min: float = 1.
	# 	# loops
	# 	printEmojis('Training neural network... 🧠')
	# 	bar_format: str =
	# 		'{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}, Elapsed: {elapsed}, ETA: {remaining}, {rate_fmt} '
	# 	with tqdm(bar_format=bar_format, total=self.P['num_of_epochs'], unit='  epochs') as epoch_bar:
	# 		with tqdm(bar_format=bar_format, total=len(training_dataset), unit=' batches') as i_bar:
	# 			with tqdm(bar_format=bar_format, total=len(testing_dataset), unit=' batches') as t_bar:
	# 				epoch = 0
	# 				while self.P['num_of_epochs'] == 0 or epoch < self.P['num_of_epochs']:
	# 					# initialise ux and loss
	# 					i_bar.reset()
	# 					t_bar.reset()
	# 					# training loop
	# 					self.M.train()
	# 					for i, (x, y) in enumerate(training_dataset):
	# 						x = x.to(self.device)
	# 						y = y.to(self.device)
	# 						y_hat = self.M(x)
	# 						loss, training_loss = self.M.loss(y, y_hat, training_loss)
	# 						assert not math.isnan(loss.item())
	# 						loss.backward()
	# 						self.M.optimiser.step()
	# 						self.M.optimiser.zero_grad()
	# 						i_bar.update(1)
	# 					# evaluation / testing
	# 					self.M.eval()
	# 					with torch.no_grad():
	# 						for t, (x, y) in enumerate(testing_dataset):
	# 							x = x.to(self.device)
	# 							y = y.to(self.device)
	# 							y_hat = self.M(x)
	# 							loss, testing_loss = self.M.loss(y, y_hat, testing_loss)
	# 							t_bar.update(1)
	# 					# calculate overall loss
	# 					training_loss = {key: value / len(training_dataset) for key, value in training_loss.items()}
	# 					testing_loss = {key: value / len(testing_dataset) for key, value in testing_loss.items()}

	# 					# if t == 0:
	# 					# plot_data: tuple[float, float] = (0., 0.)
	# 					# 	plot_data = (y.detach().cpu().numpy()[0], y_hat.detach().cpu().numpy()[0])
	# 					# 	# plots
	# 					# 	truth_fig = figure(plot_height=300, plot_width=300, title='Ground Truth')
	# 					# 	pred_fig = figure(plot_height=300, plot_width=300, title='Prediction')
	# 					# 	plot_settings = {
	# 					# 		'fill_color': '#1B9E31',
	# 					# 		'line_color': '#126B21',
	# 					# 		'x': 0.,
	# 					# 		'y': 0.,
	# 					# 	}
	# 					# 	truth_fig.circle(radius=plot_data[0] / 2, **plot_settings)
	# 					# 	pred_fig.circle(radius=plot_data[1] / 2, **plot_settings)
	# 					# 'drum_example': wandb.Html(file_html(row(truth_fig, pred_fig), CDN, 'Drum Example.')),

	# 					# save logs to wandb
	# 					if self.__using_wandb:
	# 						wandb.log({
	# 							'epoch': epoch,
	# 							'evaluation': not self.P['testing'],
	# 							'testing_loss': testing_loss,
	# 							'training_loss': training_loss,
	# 						}, commit=True)

	# 					# save model
	# 					if epoch == 0 and not os.path.isdir(self.R['model_dir']):
	# 						os.makedirs(self.R['model_dir'])
	# 					# if epoch > 50:
	# 					# 	torch.save(ExportedModel({
	# 					# 		'epoch': epoch,
	# 					# 		'evaluation_loss': testing_loss if not P['testing'] else None,
	# 					# 		'model_state_dict': self.M.state_dict(),
	# 					# 		'model_args': {
	# 					# 			'depth': P['depth'],
	# 					# 			'dropout': P['dropout'],
	# 					# 		},
	# 					# 		'model_kwargs': {},
	# 					# 		'optimizer_state_dict': optimiser.state_dict(),
	# 					# 		'testing_loss': testing_loss if P['testing'] else None,
	# 					# 		'training_loss': training_loss,
	# 					# 	}), f'{R["model_dir"]}/epoch_{epoch}.pth')
	# 					# 	if wandb_config is not None:
	# 					# 		# upload model
	# 					# 		wandb.save(
	# 					# 			os.path.join(R['model_dir'], f'epoch_{epoch}.pth'),
	# 					# 			R['model_dir'],
	# 					# 		)

	# 					# cleanup
	# 					if torch.cuda.is_available():
	# 						torch.cuda.empty_cache()
	# 					# early stopping
	# 					test_loss_arr.append(loss.item())
	# 					test_loss_arr = test_loss_arr[-32:]
	# 					test_loss_min = min(test_loss_min, loss.item())
	# 					if (min(test_loss_arr) > test_loss_min):
	# 						break
	# 					# progress bar
	# 					epoch_bar.update(1)
	# 					epoch += 1
