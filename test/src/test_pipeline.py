# core
import requests
import os
from typing import get_args
from unittest import TestCase

# dependencies
import torch	# pytorch
import wandb	# experiment tracking

# src
from kac_drumset import TestTone, TorchDataset
from kac_drumset.utils import clearDirectory, withoutPrinting
from kac_prediction.pipeline import (
	# classes
	Routine,
	Model,
	# types
	Datasets,
	Parameters,
)


class PipelineTests(TestCase):
	'''
	Tests used in conjunction with `kac_prediction/pipeline`.
	'''

	asset_dir: str = os.path.normpath(f'{os.path.dirname(__file__)}/../assets')
	tmp_dir: str = os.path.normpath(f'{os.path.dirname(__file__)}/../tmp')

	def tearDown(self) -> None:
		''' destructor '''
		clearDirectory(self.tmp_dir)

	def test_dataset_io(self) -> None:
		'''
		Tests used for dataset io.
		'''

		# This test asserts that all of the dataset endpoints in  bin/install-dataset.sh exist.
		for endpoint in get_args(Datasets):
			self.assertLess(requests.head(f'https://zenodo.org/record/7274474/files/{endpoint}.zip?download=0').status_code, 400)

	def test_local_routine(self) -> None:
		'''
		Tests used with pipeline/routines using a local run.
		'''
		# Initialise a training routine.
		with withoutPrinting():
			routine = Routine(
				exports_dir=self.tmp_dir,
				wandb_config={},
			)

		# This test asserts that the model directory exists.
		self.assertTrue(os.path.isdir(routine.R['exports_dir']))

		# This test assets that getParameters returns the correct default dict.
		routine.setParameters(
			# default parameters
			default=Parameters({
				'batch_size': 5,
				'dataset_split': (0.7, 0.15, 0.15),
				'num_of_epochs': 100,
				'testing': True,
				'with_early_stopping': False,
			}),
			config_path='',
		)
		self.assertEqual(routine.P, {
			'batch_size': 5,
			'dataset_split': (0.7, 0.15, 0.15),
			'num_of_epochs': 100,
			'testing': True,
			'with_early_stopping': False,
		})

		# This test assets that getParameters returns the correct dict with an empty config file.
		routine.setParameters(
			# default parameters
			default=Parameters({
				'batch_size': 5,
				'dataset_split': (0.7, 0.15, 0.15),
				'num_of_epochs': 100,
				'testing': True,
				'with_early_stopping': False,
			}),
			config_path=f'{self.asset_dir}/empty_config.yaml',
		)
		self.assertEqual(routine.P, {
			'batch_size': 5,
			'dataset_split': (0.7, 0.15, 0.15),
			'num_of_epochs': 100,
			'testing': True,
			'with_early_stopping': False,
		})

		# This test assets that getParameters returns the correct dict with a custom config file.
		routine.setParameters(
			# default parameters
			default=Parameters({
				'batch_size': 5,
				'dataset_split': (0.7, 0.15, 0.15),
				'num_of_epochs': 100,
				'testing': False,
				'with_early_stopping': False,
			}),
			config_path=f'{self.asset_dir}/custom_config.yaml',
		)
		self.assertEqual(routine.P, {
			'batch_size': 1,
			'dataset_split': (0.7, 0.15, 0.15),
			'num_of_epochs': 10,
			'testing': True,
			'with_early_stopping': True,
		})
		self.assertFalse(hasattr(routine.P, 'some_erroneous_key'))

		# load a simple dataset
		with withoutPrinting():
			routine.importDataset(
				dataset_dir=self.tmp_dir,
				dataset_name='',
				LocalSampler=TestTone,
			)
		routine.D.Y = torch.tensor([[y['f_0']] for y in routine.D.Y]) # type: ignore

		# This test asserts that the dataset was correctly instantiated.
		self.assertIsInstance(routine.D, TorchDataset)

		# create a simple model
		class SimpleModel(Model):
			class ModelHyperParameters(Parameters):
				pass

			def __init__(self) -> None:
				super().__init__()

			def forward(self, x: torch.Tensor) -> torch.Tensor:
				''' torch.nn.Module.forward() '''
				return x

			def innerTrainingLoop(self, i: int, loop_length: int, x: torch.Tensor, y: torch.Tensor) -> None:
				''' The training loop ran during routine.train(). '''
				self.training_loss += 1.
				if i == loop_length - 1:
					# This test asserts that the dataset was split correctly
					assert self.training_loss == loop_length and self.training_loss == 200 * 0.7

		# load model
		routine.M = SimpleModel()

		# This test asserts that the model was properly instantiated.
		self.assertIsInstance(routine.M, Model)

		# define the inner testing loop
		def innerTestingLoop(i: int, loop_length: int, x: torch.Tensor, y: torch.Tensor) -> None:
			''' The testing loop ran during routine.train(). '''
			routine.M.testing_loss += 1.
			if i == loop_length - 1:
				# This test asserts that the dataset was split correctly
				self.assertEqual(loop_length, 200 * 0.15)
				# This test asserts that, after a training run, training_loss and testing_loss are correctly at 1.
				self.assertEqual(loop_length, routine.M.testing_loss)

		# run the training routine and tests defined above.
		with withoutPrinting():
			routine.train(innerTestingLoop)

	def test_wandb_routine(self) -> None:
		'''
		Tests used with pipeline/routines using a wandb run.
		'''
		# Run a training routine.
		with withoutPrinting():
			routine = Routine(
				exports_dir='',
				wandb_config={
					'entity': 'lewiswolf',
					'project': 'liltester',
				},
			)
			wandb.finish()

		# This test asserts that the model directory exists.
		self.assertTrue(os.path.isdir(routine.R['exports_dir']))
