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
from kac_prediction.pipeline import importDataset, Routine

__all__ = ['SizeOfCircularDrum']


class SizeOfCircularDrum(Routine):
	'''
	'''

	def __init__(
		self,
		config_path: str = '',
		testing: bool = True,
		wandb_config: dict[str, Any] = {},
	) -> None:
		'''
		Perform the entire training routine
		'''

		# initialise a default run
		super().__init__()
		self.R = self.getRunInfo(
			model_dir=os.path.normpath(
				f'{os.path.dirname(__file__)}/../model',
			),
			wandb_config=wandb_config,
		)

		# initialise parameters
		self.P = self.getParams(
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
		self.D = importDataset(
			dataset_dir=os.path.normpath(f'{os.path.dirname(__file__)}/../data'),
			dataset_name='' if self.P['testing'] else '5000-circular-drums-of-varying-size',
			LocalSampler=BesselModel,
			representation_settings=RepresentationSettings({
				'normalise_input': True,
				'output_type': 'end2end',
			}),
		)
		self.D.X = torch.narrow(self.D.X, 1, 0, 1024)
		self.D.Y = torch.tensor([[y['drum_size']] for y in self.D.Y]) # type: ignore

		# configure model
		self.M = CRePE(
			depth=self.P['depth'],
			dropout=self.P['dropout'],
			learning_rate=self.P['learning_rate'],
			optimiser=self.P['optimiser'],
			outputs=1,
		).to(self.device)

		# self.train()


if __name__ == '__main__':
	R = SizeOfCircularDrum(wandb_config={
		'entity': 'lewiswolf',
		'project': 'kac_prediction (circular drum size)',
	})
	exit()
