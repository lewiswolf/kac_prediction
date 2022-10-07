'''
Full training and evaluation routine for the SizeOfCircularDrum model.
'''

# core
import argparse
import os
import random
# import shlex
import string
# import subprocess
from typing import Any, Literal, TypedDict
import yaml

# dependencies
import torch			# pytorch
from tqdm import tqdm	# progress bar
import wandb			# experiment tracking

# src
from kac_drumset import (
	# methods
	generateDataset,
	loadDataset,
	transformDataset,
	# Classes
	BesselModel,
	# types
	RepresentationSettings,
)
from kac_drumset.utils import printEmojis
from ..architecture.size_of_circular_drum import SizeOfCircularDrum
from ..pipeline import (
	getEvaluationDataset,
	getTestingDataset,
	getTrainingDataset,
)
from ..pipeline.types import RunInfo


# types
class ModelHyperParameters(TypedDict):
	''' Hyper parameters for this specific model. '''
	batch_size: int
	depth: Literal['large', 'medium', 'small', 'tiny']
	dropout: float
	learning_rate: float
	num_of_epochs: int
	optimiser: Literal['adam', 'sgd']
	testing: bool


# initialise arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', default=None, help='yaml config file.')
parser.add_argument('--wandb', action='store_true', help='log to wandb')
args = parser.parse_args()

# load a yaml config file for a single run
Y: dict[str, Any] = {}
if args.config is not None:
	with open(args.config, 'r') as f:
		Y.update(yaml.safe_load(f))

# initialise hyper parameters
P: ModelHyperParameters = {
	'batch_size': Y['batch_size'] if 'batch_size' in Y else 5,
	'depth': 'tiny',
	'dropout': Y['dropout'] if 'dropout' in Y else 0.25,
	'learning_rate': Y['learning_rate'] if 'learning_rate' in Y else 1e-3,
	'num_of_epochs': Y['num_of_epochs'] if 'num_of_epochs' in Y else 100,
	'optimiser': Y['optimiser'] if 'optimiser' in Y else 'sgd',
	'testing': Y['testing'] if 'testing' in Y else True,
}

# initialise a local run as default
i_d = ''.join(random.choice(string.ascii_letters) for x in range(10))
run: RunInfo = {
	'id': i_d,
	'model_dir': os.path.normpath(f'{os.path.dirname(__file__)}/../../model/run_{i_d}'),
}

# initialise weights and biases
if args.wandb:
	wandb.login()
	wandb.init(project="kac_prediction (circular drum size)", entity="lewiswolf")
	P.update({
		'batch_size': wandb.config['batch_size'] if 'batch_size' in wandb.config else P['batch_size'],
		'dropout': wandb.config['dropout'] if 'dropout' in wandb.config else P['dropout'],
		'learning_rate': wandb.config['learning_rate'] if 'learning_rate' in wandb.config else P['learning_rate'],
		'num_of_epochs': wandb.config['num_of_epochs'] if 'num_of_epochs' in wandb.config else P['num_of_epochs'],
		'optimiser': wandb.config['optimiser'] if 'optimiser' in wandb.config else P['optimiser'],
	})
	wandb.config.update(P)
	if wandb.run is not None:
		run.update({
			'id': wandb.run.id,
			'model_dir': wandb.run.dir,
		})

# load, generate or install a dataset
dataset_dir: str = os.path.normpath(f'{os.path.dirname(__file__)}/../../data')
representation_settings: RepresentationSettings = {
	'normalise_input': True,
	'output_type': 'end2end',
}
try:
	dataset = transformDataset(loadDataset(dataset_dir=dataset_dir), representation_settings)
except Exception as e:
	if type(e).__name__ == 'FileNotFoundError':
		# import the official dataset for this project
		# subprocess.run(shlex.split('sh ./bin/install-dataset.sh 5000-circular-drums-of-varying-size'))
		# dataset = transformDataset(loadDataset(dataset_dir=dataset_dir), representation_settings)
		# generate a dataset locally
		dataset = generateDataset(
			BesselModel,
			dataset_dir=dataset_dir,
			dataset_size=500,
			representation_settings=representation_settings,
			sampler_settings=BesselModel.Settings({
				'duration': 1.,
				'sample_rate': 48000,
			}),
		)
	else:
		raise e
training_dataset = getTrainingDataset(dataset, batch_size=P['batch_size'])
testing_dataset = getTestingDataset(dataset) if P['testing'] else getEvaluationDataset(dataset)

# configure device
if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	printEmojis('😓 WARNING 😓 Nvidia GPU support is not available for training the network.')
	device = torch.device('cpu')

# configure model
model = SizeOfCircularDrum(P['depth'], P['dropout']).to(device)
if args.wandb:
	wandb.watch(model, log_freq=1000)

# configure criterion
criterion = torch.nn.MSELoss()

# configure optimiser
optimiser: torch.optim.Optimizer | None = None
if P['optimiser'] == 'adam':
	optimiser = torch.optim.Adam(model.parameters(), lr=P['learning_rate'])
elif P['optimiser'] == 'sgd':
	optimiser = torch.optim.SGD(model.parameters(), lr=P['learning_rate'])

# loops
printEmojis('Training neural network... 🧠')
bar_format: str = '{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}, Elapsed: {elapsed}, ETA: {remaining}, {rate_fmt} '
with tqdm(bar_format=bar_format, total=P['num_of_epochs'], unit='epochs') as epoch_bar:
	with tqdm(bar_format=bar_format, total=len(training_dataset), unit='batches') as i_bar:
		with tqdm(bar_format=bar_format, total=len(testing_dataset), unit='samples') as t_bar:
			for epoch in range(P['num_of_epochs']):

				# initialise ux and loss
				i_bar.reset()
				t_bar.reset()
				testing_loss = 0.
				training_loss = 0.

				# training loop
				model.train()
				for i, (x, y) in enumerate(training_dataset):
					x = x.to(device)
					y = y['drum_size'].to(device)
					y_hat = model(x)
					loss = criterion(y_hat, y)
					loss.backward()
					training_loss += loss.item()
					optimiser.step()
					optimiser.zero_grad()
					i_bar.update(1)

				# evaluation / testing
				model.eval()
				with torch.no_grad():
					for t, (x, y) in enumerate(testing_dataset):
						x = x.to(device)
						y = y['drum_size'].to(device)
						y_hat = model(x)
						testing_loss += criterion(y_hat, y).item()
						t_bar.update(1)

				# calculate overall loss
				testing_loss /= len(testing_dataset)
				training_loss /= len(training_dataset)

				if args.wandb:
					# logs
					wandb.log({
						'epoch': epoch,
						'evaluation_loss': testing_loss if not P['testing'] else None,
						'testing_loss': testing_loss if P['testing'] else None,
						'training_loss': training_loss,
					}, commit=True)
					# save model
					wandb.save(
						os.path.join(run['model_dir'], f'epoch_{epoch}.Model'),
						run['model_dir'],
					)
				else:
					# save model locally
					if epoch == 0 and not os.path.isdir(run['model_dir']):
						os.makedirs(run['model_dir'])
					torch.save({
						'epoch': epoch,
						'loss': loss,
						'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimiser.state_dict(),
					}, f'{run["model_dir"]}/epoch_{epoch}.Model')

				# cleanup
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
				epoch_bar.update(1)
if args.wandb:
	wandb.finish()
