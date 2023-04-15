# core
import requests
import os
from typing import get_args
from unittest import TestCase

# src
from kac_drumset.utils import clearDirectory
from kac_prediction.pipeline import (
	inferDatasetSplit,
	Types as T,
)


class PipelineTests(TestCase):
	'''
	Tests used in conjunction with `kac_prediction/pipeline`.
	'''

	tmp_dir: str = os.path.normpath(f'{os.path.dirname(__file__)}/../tmp')

	def tearDown(self) -> None:
		''' destructor '''
		clearDirectory(self.tmp_dir)

	def test_dataset_io(self) -> None:
		'''
		Tests used for dataset io.
		'''

		# This test asserts that all of the dataset endpoints in  bin/install-dataset.sh exist.
		for endpoint in get_args(T.Datasets):
			self.assertLess(requests.head(f'https://zenodo.org/record/7274474/files/{endpoint}.zip?download=0').status_code, 400)

		# This test asserts that inferDatasetSplit returns the correct values.
		self.assertEqual(inferDatasetSplit(10), (6, 8, 10))
		self.assertEqual(inferDatasetSplit(11), (7, 9, 11))
		self.assertEqual(inferDatasetSplit(12), (8, 10, 12))
		self.assertEqual(inferDatasetSplit(13), (9, 11, 13))
		self.assertEqual(inferDatasetSplit(14), (10, 12, 14))
		self.assertEqual(inferDatasetSplit(15), (11, 13, 15))
		self.assertEqual(inferDatasetSplit(16), (12, 14, 16))
		self.assertEqual(inferDatasetSplit(17), (11, 14, 17))
		self.assertEqual(inferDatasetSplit(18), (12, 15, 18))
		self.assertEqual(inferDatasetSplit(19), (13, 16, 19))
		self.assertEqual(inferDatasetSplit(20), (14, 17, 20))
