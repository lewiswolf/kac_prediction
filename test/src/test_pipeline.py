# core
import requests
import os
from typing import get_args
from unittest import TestCase

# dependencies
import wandb			# experiment tracking

# src
from kac_drumset.utils import clearDirectory, withoutPrinting
from kac_prediction.pipeline import (
	Routine,
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

		# This test asserts that inferDatasetSplit returns the correct values.
		# self.assertEqual(inferDatasetSplit(10), (6, 8, 10))
		# self.assertEqual(inferDatasetSplit(11), (7, 9, 11))
		# self.assertEqual(inferDatasetSplit(12), (8, 10, 12))
		# self.assertEqual(inferDatasetSplit(13), (9, 11, 13))
		# self.assertEqual(inferDatasetSplit(14), (10, 12, 14))
		# self.assertEqual(inferDatasetSplit(15), (11, 13, 15))
		# self.assertEqual(inferDatasetSplit(16), (12, 14, 16))
		# self.assertEqual(inferDatasetSplit(17), (11, 14, 17))
		# self.assertEqual(inferDatasetSplit(18), (12, 15, 18))
		# self.assertEqual(inferDatasetSplit(19), (13, 16, 19))
		# self.assertEqual(inferDatasetSplit(20), (14, 17, 20))

	def test_local_routine(self) -> None:
		'''
		Tests used with pipeline/routines using a local run.
		'''
		# Initialise a training routine.
		with withoutPrinting():
			routine = Routine(
				model_dir=self.tmp_dir,
				wandb_config={},
			)

		# This test asserts that the model directory exists.
		self.assertTrue(os.path.isdir(routine.R['model_dir']))

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
				'testing': True,
				'with_early_stopping': False,
			}),
			config_path=f'{self.asset_dir}/custom_config.yaml',
		)
		self.assertEqual(routine.P, {
			'batch_size': 10,
			'dataset_split': (0.7, 0.15, 0.15),
			'num_of_epochs': 1000,
			'testing': False,
			'with_early_stopping': True,
		})
		self.assertFalse(hasattr(routine.P, 'some_erroneous_key'))

	def test_wandb_routine(self) -> None:
		'''
		Tests used with pipeline/routines using a wandb run.
		'''
		# Run a training routine.
		with withoutPrinting():
			routine = Routine(
				model_dir='',
				wandb_config={
					'entity': 'lewiswolf',
					'project': 'liltester',
				},
			)
			wandb.finish()

		# This test asserts that the model directory exists.
		self.assertTrue(os.path.isdir(routine.R['model_dir']))
