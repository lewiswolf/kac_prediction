'''
Full training and evaluation routine for the SizeOfCircularDrum model.
'''

# core
import os
from typing import Any

# dependencies
import torch		# pytorch

# src
from kac_drumset import BesselModel, RepresentationSettings
from kac_prediction.architecture import CRePE
from kac_prediction.pipeline import Routine

__all__ = ['SizeOfCircularDrum']


def SizeOfCircularDrum(config_path: str = '', testing: bool = True, wandb_config: dict[str, Any] = {}) -> None:
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
			'dataset_split': (0.7, 0.15, 0.15),
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
		dataset_dir=os.path.normpath(f'{os.path.dirname(__file__)}/../../data'),
		dataset_name='' if routine.P['testing'] else '5000-circular-drums-of-varying-size',
		LocalSampler=BesselModel,
		representation_settings=RepresentationSettings({'normalise_input': True, 'output_type': 'end2end'}),
		sampler_settings=BesselModel.Settings({'duration': 1., 'sample_rate': 48000}),
	)
	routine.D.X = torch.narrow(routine.D.X, 1, 0, 1024)
	routine.D.Y = torch.tensor([[y['drum_size']] for y in routine.D.Y]) # type: ignore

	# configure model
	routine.M = CRePE(
		depth=routine.P['depth'],
		dropout=routine.P['dropout'],
		learning_rate=routine.P['learning_rate'],
		optimiser=routine.P['optimiser'],
		outputs=1,
	).to(routine.device)

	# self.train()


if __name__ == '__main__':
	SizeOfCircularDrum(wandb_config={
		'entity': 'lewiswolf',
		'project': 'kac_prediction (circular drum size)',
	})
	exit()
