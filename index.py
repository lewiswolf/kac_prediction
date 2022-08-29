# src
from kac_drumset import (
	# methods
	generateDataset,
	loadDataset,
	transformDataset,
	# Classes
	FDTDModel,
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
	representation_settings: RepresentationSettings = {'output_type': 'end2end'}

	# load or generate a dataset
	try:
		dataset = transformDataset(loadDataset(), representation_settings)
	except Exception as e:
		if type(e).__name__ == 'FileNotFoundError':
			dataset = generateDataset(
				FDTDModel,
				dataset_size=10,
				representation_settings=representation_settings,
				sampler_settings=FDTDModel.Settings({
					'duration': 1.0,
					'sample_rate': 48000,
				}),
			)
	# train model
	trainModel(*getTrainingDatasets(dataset, batch_size))
	exit()
