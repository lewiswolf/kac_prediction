# core
import os

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
from kac_prediction.pipeline import (
	getTrainingDatasets,
	trainModel,
)


if __name__ == '__main__':
	# settings
	batch_size: int = 4
	dataset_dir = os.path.normpath(f'{os.path.dirname(__file__)}/data')
	representation_settings: RepresentationSettings = {'output_type': 'end2end'}

	# load or generate a dataset
	try:
		dataset = transformDataset(loadDataset(dataset_dir=dataset_dir), representation_settings)
	except Exception as e:
		if type(e).__name__ == 'FileNotFoundError':
			dataset = generateDataset(
				BesselModel,
				dataset_dir=dataset_dir,
				dataset_size=1000,
				representation_settings=representation_settings,
				sampler_settings=BesselModel.Settings({
					'duration': 1.0,
					'sample_rate': 48000,
				}),
			)
		else:
			raise e
	# train model
	trainModel(*getTrainingDatasets(dataset, batch_size))
	exit()
