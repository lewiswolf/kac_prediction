'''
Full training and evaluation routine for the SizeOfCircularDrum model.
'''

# core
import os
from typing import Any

# dependencies
import torch						# pytorch
import wandb						# experiment tracking

# src
from kac_drumset import PoissonModel, RepresentationSettings
from kac_prediction.architecture import CRePE
from kac_prediction.pipeline import Routine

__all__ = ['DimOfRectangularDrum']


def DimOfRectangularDrum(config_path: str = '', testing: bool = True, wandb_config: dict[str, Any] = {}) -> None:
	'''
	Perform the entire training routine
	'''

	# initialise a default routine
	routine = Routine(
		exports_dir=os.path.normpath(f'{os.path.dirname(__file__)}/../../model'),
		wandb_config=wandb_config,
	)

	# initialise parameters
	routine.setParameters(
		# default parameters
		CRePE.ModelHyperParameters({
			'batch_size': 5,
			'dataset_split': (0.7, 0.15, 0.15),
			'depth': 'tiny',
			'dropout': 0.25,
			'learning_rate': 1e-3,
			'num_of_epochs': 100,
			'optimiser': 'sgd',
			'testing': testing,
			'with_early_stopping': False,
		}),
		# yaml config path
		config_path=config_path,
	)

	# load, generate or install a dataset
	routine.importDataset(
		dataset_dir=os.path.normpath(f'{os.path.dirname(__file__)}/../../data'),
		dataset_name='' if routine.P['testing'] else '5000-rectangular-drums-of-varying-dimension',
		LocalSampler=PoissonModel,
		representation_settings=RepresentationSettings({'normalise_input': True, 'output_type': 'end2end'}),
		sampler_settings=PoissonModel.Settings({'duration': 1., 'sample_rate': 48000}),
	)
	# shape data
	routine.D.X = torch.narrow(routine.D.X, 1, 0, 1024)
	routine.D.Y = torch.tensor([[y['drum_size']] for y in routine.D.Y]) # type: ignore

	# configure model
	routine.M = CRePE(
		depth=routine.P['depth'],
		dropout=routine.P['dropout'],
		learning_rate=routine.P['learning_rate'],
		optimiser=routine.P['optimiser'],
		outputs=2,
	)

	# define how the model is to be tested
	def innerTestingLoop(i: int, loop_length: float, x: torch.Tensor, y: torch.Tensor) -> None:
		'''
		This method should be designed to satisfy the loop:
			for i, (x, y) in enumerate(testing_dataset):
				Model.innerTrainingLoop(i, len(testing_dataset), x.to(device), y.to(device))
		and should somewhere include the line:
			self.testing_loss += ...
		'''
		y_hat = routine.M(x)
		routine.M.testing_loss += routine.M.criterion(y, y_hat).item() / loop_length
		# plots
		if routine.using_wandb and i == loop_length - 1:
			# logs
			wandb.log({
				'epoch': routine.epoch,
				'evaluation_loss': routine.M.testing_loss if not routine.P['testing'] else None,
				'testing_loss': routine.M.testing_loss if routine.P['testing'] else None,
				'training_loss': routine.M.training_loss,
			}, commit=True)

	# train and test a model
	routine.train(innerTestingLoop)


if __name__ == '__main__':
	DimOfRectangularDrum(wandb_config={
		'entity': 'lewiswolf',
		'project': 'kac_prediction (rectangular drum dimension)',
	})
	exit()
