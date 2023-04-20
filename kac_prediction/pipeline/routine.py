'''
'''

# core
from itertools import accumulate
import os
import random
import shlex
import string
import subprocess
from typing import Any, Literal
import yaml

# dependencies
import torch			# pytorch
from tqdm import tqdm	# progress bar
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
				assert dataset_name != '' or LocalSampler is not None, \
					'importDataset requires at least a dataset_name or a LocalSampler to generate a new dataset'
				# import the official dataset for this project
				if dataset_name != '':
					subprocess.run(shlex.split(f'sh ./bin/install-dataset.sh {dataset_name}'))
					dataset = transformDataset(loadDataset(dataset_dir=dataset_dir), representation_settings)
				# generate a dataset locally
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
		# handle errors
		assert hasattr(self, 'D'), 'Routine.D: TorchDataset is not set.'
		assert hasattr(self, 'M'), 'Routine.M: Model is not set.'
		assert hasattr(self, 'P'), 'Routine.P: Parameters is not set. Run Routine.getParameters()'
		assert hasattr(self, 'R'), 'Routine.R: RunInfo is not set. Run Routine.getRunInfo()'

		# split dataset
		subdivisions = [round(self.D.__len__() * p) for p in self.P['dataset_split']]
		# this correction supposes that split[0] > split[1 or 2]
		subdivisions[0] += self.D.__len__() - sum(subdivisions)
		subdivisions = list(accumulate(subdivisions))[0:3]
		training_dataset = torch.utils.data.DataLoader(
			self.D,
			batch_size=self.P['batch_size'],
			sampler=torch.utils.data.SubsetRandomSampler(list(range(0, subdivisions[0]))),
		)
		testing_dataset = torch.utils.data.DataLoader(
			self.D,
			batch_size=self.P['batch_size'],
			sampler=torch.utils.data.SubsetRandomSampler(list(range(subdivisions[0], subdivisions[1]))),
		) if self.P['testing'] else torch.utils.data.DataLoader(
			self.D,
			batch_size=self.P['batch_size'],
			sampler=torch.utils.data.SubsetRandomSampler(list(range(subdivisions[1], subdivisions[2]))),
		)
		# # init loss
		# training_loss: dict[str, float | int] = {}
		# testing_loss: dict[str, float | int] = {}
		# test_loss_arr: list[float] = []
		# test_loss_min: float = 1.
		# loops
		printEmojis('Training neural network... 🧠')
		if self._using_wandb:
			wandb.watch(self.M, log_freq=1000)
		bar_format: str = '{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}, Elapsed: {elapsed}, ETA: {remaining}, {rate_fmt} '
		with tqdm(bar_format=bar_format, total=self.P['num_of_epochs'], unit='  epochs') as epoch_bar:
			with tqdm(bar_format=bar_format, total=len(training_dataset), unit=' batches') as i_bar:
				with tqdm(bar_format=bar_format, total=len(testing_dataset), unit=' batches') as t_bar:
					epoch = 0
					while self.P['num_of_epochs'] == 0 or epoch < self.P['num_of_epochs']:
						# initialise ux and loss
						i_bar.reset()
						t_bar.reset()
		# 				# training loop
		# 				self.M.train()
		# 				for (x, y) in training_dataset:
		# 					x = x.to(self.device)
		# 					y = y.to(self.device)
		# 					y_hat = self.M(x)
		# 					loss, training_loss = self.M.loss(y, y_hat, training_loss)
		# 					assert not math.isnan(loss.item())
		# 					loss.backward()
		# 					self.M.optimiser.step()
		# 					self.M.optimiser.zero_grad()
		# 					i_bar.update(1)
		# 				# evaluation / testing
		# 				self.M.eval()
		# 				with torch.no_grad():
		# 					for (x, y) in testing_dataset:
		# 						x = x.to(self.device)
		# 						y = y.to(self.device)
		# 						y_hat = self.M(x)
		# 						loss, testing_loss = self.M.loss(y, y_hat, testing_loss)
		# 						t_bar.update(1)
		# 				# calculate overall loss
		# 				training_loss = {key: value / len(training_dataset) for key, value in training_loss.items()}
		# 				testing_loss = {key: value / len(testing_dataset) for key, value in testing_loss.items()}

						# if t == 0:
						# plot_data: tuple[float, float] = (0., 0.)
						# 	plot_data = (y.detach().cpu().numpy()[0], y_hat.detach().cpu().numpy()[0])
						# 	# plots
						# 	truth_fig = figure(plot_height=300, plot_width=300, title='Ground Truth')
						# 	pred_fig = figure(plot_height=300, plot_width=300, title='Prediction')
						# 	plot_settings = {
						# 		'fill_color': '#1B9E31',
						# 		'line_color': '#126B21',
						# 		'x': 0.,
						# 		'y': 0.,
						# 	}
						# 	truth_fig.circle(radius=plot_data[0] / 2, **plot_settings)
						# 	pred_fig.circle(radius=plot_data[1] / 2, **plot_settings)
						# 'drum_example': wandb.Html(file_html(row(truth_fig, pred_fig), CDN, 'Drum Example.')),

						# save logs to wandb
						# if self._using_wandb:
						# 	wandb.log({
						# 		'epoch': epoch,
						# 		'evaluation': not self.P['testing'],
						# 		'testing_loss': testing_loss,
						# 		'training_loss': training_loss,
						# 	}, commit=True)

						# save model
						# if epoch > 50:
						# 	torch.save(ExportedModel({
						# 		'epoch': epoch,
						# 		'evaluation_loss': testing_loss if not P['testing'] else None,
						# 		'model_state_dict': self.M.state_dict(),
						# 		'model_args': {
						# 			'depth': P['depth'],
						# 			'dropout': P['dropout'],
						# 		},
						# 		'model_kwargs': {},
						# 		'optimizer_state_dict': optimiser.state_dict(),
						# 		'testing_loss': testing_loss if P['testing'] else None,
						# 		'training_loss': training_loss,
						# 	}), f'{R["model_dir"]}/epoch_{epoch}.pth')
						# 	if wandb_config is not None:
						# 		# upload model
						# 		wandb.save(
						# 			os.path.join(R['model_dir'], f'epoch_{epoch}.pth'),
						# 			R['model_dir'],
						# 		)

						# early stopping
						# test_loss_arr.append(loss.item())
						# test_loss_arr = test_loss_arr[-32:]
						# test_loss_min = min(test_loss_min, loss.item())
						# if (min(test_loss_arr) > test_loss_min):
						# 	break
						# cleanup
						if torch.cuda.is_available():
							torch.cuda.empty_cache()
						# progress bar
						epoch_bar.update(1)
						epoch += 1
