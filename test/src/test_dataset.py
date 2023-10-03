# core
from importlib.metadata import version
import os
from unittest import TestCase

# dependencies
import numpy as np 				# maths
import torch					# pytorch

# src
from kac_prediction.dataset import (
	InputRepresentation,
	generateDataset,
	loadDataset,
	regenerateDataPoints,
	transformDataset,
)
from kac_prediction.samplers import TestSweep, TestTone
from kac_prediction.utils import clearDirectory
from .utils import withoutPrinting


class DatasetTests(TestCase):
	'''
	Tests used in conjunction with `/dataset`.
	'''

	tmp_dir: str = os.path.normpath(f'{os.path.dirname(__file__)}/../tmp')
	tone = TestSweep(1., 48000)

	def tearDown(self) -> None:
		''' destructor '''
		clearDirectory(self.tmp_dir)

	def test_generate_dataset(self) -> None:
		'''
		Tests used in conjunction with `/dataset/generate_dataset.py`.
		'''

		# This test asserts that dynamic typing works for sampler_settings.
		with withoutPrinting():
			dataset = generateDataset(
				TestTone,
				dataset_dir=self.tmp_dir,
				dataset_size=10,
				sampler_settings=TestTone.Settings({
					'duration': 1.,
					'f_0': 440.,
					'sample_rate': 48000,
				}),
			)
			# This test asserts that the dataset retains all available information about the sampler.
			self.assertEqual(1., dataset.sampler_settings['duration'])
			self.assertEqual(440., dataset.sampler_settings['f_0'])
			self.assertEqual(48000, dataset.sampler_settings['sample_rate'])
			self.assertEqual('sin', dataset.sampler_settings['waveshape'])

		# Generate a dataset for subsequent tests.
		with withoutPrinting():
			dataset = generateDataset(
				TestSweep,
				dataset_dir=self.tmp_dir,
				dataset_size=10,
				sampler_settings=TestSweep.Settings({
					'duration': 1.,
					'sample_rate': 48000,
				}),
			)

		# This test asserts that the dataset is the correct size, both in memory and on disk.
		self.assertEqual(dataset.__len__(), 10)
		self.assertEqual(len(dataset.Y), 10)
		self.assertEqual(len(os.listdir(f'{os.getcwd()}/test/tmp')) - 2, 10)

		# This test asserts that the data is the correct data type.
		self.assertEqual(dataset.X.dtype, torch.float64)
		for v in dataset.Y[0].values():
			self.assertEqual(v.dtype, torch.float64)

		# This test asserts that the dataset directory is correct.
		self.assertEqual(dataset.dataset_dir, self.tmp_dir)

		# This test asserts that the sampler information is correct.
		self.assertEqual(dataset.sampler['name'], 'TestSweep')
		self.assertEqual(type(dataset.sampler['version']), str)
		self.assertEqual(dataset.sampler['version'], version('kac_prediction'))

		# This test asserts that the SamplerSettings were copied correctly.
		self.assertEqual(dataset.sampler_settings, {
			'duration': 1.,
			'sample_rate': 48000,
		})

	def test_input_representation(self) -> None:
		'''
		Tests used in conjunction with `/dataset/input_representation.py`.
		'''

		# This test asserts that a normalised waveform is always bounded by [-1., 1.].
		norm = InputRepresentation.normalise(self.tone.waveform)
		self.assertEqual(norm.max(), 1.)
		self.assertEqual(norm.min(), -1.)

		# This test asserts that a DC signal can be normalised
		norm = InputRepresentation.normalise(np.ones(100))
		self.assertEqual(norm.max(), 0.)
		self.assertEqual(norm.min(), 0.)

		# Test an end to end input representation.
		IR = InputRepresentation(self.tone.sample_rate, {
			'normalise_input': False,
			'output_type': 'end2end',
		})
		representation = IR.transform(self.tone.waveform)

		# This test asserts that the RepresentationSettings are as expected.
		test_settings = {
			'f_min': 22.05,
			'hop_length': 256,
			'n_bins': 512,
			'n_mels': 32,
			'normalise_input': False,
			'output_type': 'end2end',
			'window_length': 512,
		}
		self.assertEqual(IR.settings, test_settings)

		# This test asserts that the input waveform and the transform are equivalent.
		self.assertTrue(np.array_equal(self.tone.waveform, representation.detach().numpy()))

		# This test asserts that the output tensor is the correct shape and type.
		self.assertEqual(representation.shape, IR.transformShape(self.tone.length, IR.settings))
		self.assertEqual(representation.dtype, torch.float64)

		# Test an FFT input representation.
		IR = InputRepresentation(self.tone.sample_rate, {'output_type': 'fft'})
		representation = IR.transform(self.tone.waveform)

		# This test asserts that the RepresentationSettings are as expected.
		test_settings['normalise_input'] = True
		test_settings['output_type'] = 'fft'
		self.assertEqual(IR.settings, test_settings)

		# This test asserts that the output tensor is the correct shape and type.
		self.assertEqual(representation.shape, IR.transformShape(self.tone.length, IR.settings))
		self.assertEqual(representation.dtype, torch.float64)

		# Test a Mel input representation.
		IR = InputRepresentation(self.tone.sample_rate, {'output_type': 'mel'})
		representation = IR.transform(self.tone.waveform)

		# This test asserts that the RepresentationSettings are as expected.
		test_settings['output_type'] = 'mel'
		self.assertEqual(IR.settings, test_settings)

		# This test asserts that the output tensor is the correct shape and type.
		self.assertEqual(representation.shape, IR.transformShape(self.tone.length, IR.settings))
		self.assertEqual(representation.dtype, torch.float64)

	def test_load_dataset(self) -> None:
		'''
		Tests used in conjunction with `/dataset/load_dataset.py`.
		'''

		# Generate and load a dataset for subsequent tests.
		with withoutPrinting():
			generateDataset(
				TestSweep,
				dataset_dir=self.tmp_dir,
				dataset_size=10,
				sampler_settings=TestSweep.Settings({
					'duration': 1.,
					'sample_rate': 48000,
				}),
			)
			dataset = loadDataset(dataset_dir=self.tmp_dir)

		# This test asserts that the dataset directory is correct.
		self.assertEqual(dataset.dataset_dir, self.tmp_dir)

		# This test asserts that the sampler name is correct.
		self.assertEqual(dataset.sampler['name'], 'TestSweep')

	def test_regenerate_dataset(self) -> None:
		'''
		Tests used in conjunction with `/dataset/regenerate_data_points.py`.
		'''

		# Generate a dataset for subsequent tests.
		with withoutPrinting():
			dataset = generateDataset(
				TestTone,
				dataset_dir=self.tmp_dir,
				dataset_size=10,
				sampler_settings=TestTone.Settings({
					'duration': 1.,
					'sample_rate': 48000,
				}),
			)

		# This test assets that the respective dataset entries get properly updated.
		old_dataset = [dataset.__getitem__(i)[1]['f_0'] for i in range(dataset.__len__())]
		with withoutPrinting():
			regenerateDataPoints(dataset, TestTone, [0])
		self.assertNotEqual(
			old_dataset[0],
			dataset.__getitem__(0)[1]['f_0'],
		)
		for i in range(1, dataset.__len__()):
			self.assertEqual(
				old_dataset[i],
				dataset.__getitem__(i)[1]['f_0'],
			)

	def test_transform_dataset(self) -> None:
		'''
		Tests used in conjunction with `/dataset/transform_dataset.py`.
		'''

		# Generate a dataset for subsequent tests.
		with withoutPrinting():
			dataset = generateDataset(
				TestSweep,
				dataset_dir=self.tmp_dir,
				dataset_size=10,
				sampler_settings=TestSweep.Settings({
					'duration': 1.,
					'sample_rate': 48000,
				}),
			)

		# This test asserts that the representation_settings are the default.
		self.assertEqual(dataset.representation_settings['output_type'], 'end2end')

		# This test asserts that data is the expected shape.
		self.assertEqual(
			dataset.__getitem__(0)[0].shape,
			InputRepresentation.transformShape(
				48000,
				dataset.representation_settings,
			),
		)

		# Transform a dataset for subsequent tests.
		with withoutPrinting():
			dataset = transformDataset(dataset, {'output_type': 'fft'})

		# This test asserts that the representation_settings have been updated.
		self.assertEqual(dataset.representation_settings['output_type'], 'fft')

		# This test asserts that data is the expected shape.
		self.assertEqual(
			dataset.__getitem__(0)[0].shape,
			InputRepresentation.transformShape(
				48000,
				dataset.representation_settings,
			),
		)

		# This test asserts that the dataset directory is correct.
		self.assertEqual(dataset.dataset_dir, self.tmp_dir)

		# This test asserts that the sampler name is correct.
		self.assertEqual(dataset.sampler['name'], 'TestSweep')

		# This test asserts that the transformed dataset can still be loaded.
		with withoutPrinting():
			dataset = loadDataset(dataset_dir=self.tmp_dir)
