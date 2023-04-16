'''
Full training and evaluation routine for the DimOfRectangularDrum model.
'''

# core
import os
from typing import Any

# dependencies
import torch		# pytorch

# src
from kac_drumset import PoissonModel, RepresentationSettings
from kac_prediction.architecture import CRePE
from kac_prediction.pipeline import Routine

__all__ = ['DimOfRectangularDrum']


def DimOfRectangularDrum(config_path: str = '', testing: bool = True, wandb_config: dict[str, Any] = {}) -> None:
	'''
	Perform the entire training routine
	'''

	# initialise a default run
	routine = Routine(
		model_dir=os.path.normpath(f'{os.path.dirname(__file__)}/../../model'),
		wandb_config=wandb_config,
	)

	# initialise parameters
	routine.setParameters(
		# default parameters
		CRePE.ModelHyperParameters({
			'batch_size': 5,
			'depth': 'tiny',
			'dropout': 0.25,
			'learning_rate': 1e-3,
			'num_of_epochs': 100,
			'optimiser': 'sgd',
			'testing': testing,
			'with_early_stopping': False,
		}),
		config_path=config_path,
	)

	# load, generate or install a dataset
	routine.importDataset(
		dataset_dir=os.path.normpath(f'{os.path.dirname(__file__)}/../data'),
		dataset_name='' if routine.P['testing'] else '5000-rectangular-drums-of-varying-dimension',
		LocalSampler=PoissonModel,
		representation_settings=RepresentationSettings({
			'normalise_input': True,
			'output_type': 'end2end',
		}),
	)
	routine.D.X = torch.narrow(routine.D.X, 1, 0, 1024)
	routine.D.Y = torch.tensor([[y['drum_size']] for y in routine.D.Y]) # type: ignore

	# configure model
	routine.M = CRePE(
		depth=routine.P['depth'],
		dropout=routine.P['dropout'],
		learning_rate=routine.P['learning_rate'],
		optimiser=routine.P['optimiser'],
		outputs=2,
	).to(routine.device)

	# self.train()


if __name__ == '__main__':
	DimOfRectangularDrum(wandb_config={
		'entity': 'lewiswolf',
		'project': 'kac_prediction (rectangular drum dimension)',
	})
	exit()
