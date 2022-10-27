'''
A simple wrapper for `pipenv run start`. This file takes advantage of the fact scripts are run upon import.
'''

if __name__ == '__main__':

	'''
	Training routine.
	'''

	# core
	import argparse

	# src
	# switch the import statement to a different routine to work with a different model.
	import routines.size_of_circular_drum as model # noqa: F401

	# initialise arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default=None, help='yaml config file.')
	parser.add_argument('--wandb', action='store_true', help='log to wandb')
	args = parser.parse_args()

	# train model
	model.train(config=args.config, using_wandb=args.wandb)
	exit()

	'''
	Deploying.
	'''

	# # core
	# import os

	# # src
	# from kac_drumset import loadDataset
	# from kac_prediction.architecture import SizeOfCircularDrum
	# from kac_prediction.pipeline import loadModel

	# # load model
	# path_to_parameters = os.path.normpath(f'{__file__}/../model/parameters.model')
	# model = loadModel(
	# 	SizeOfCircularDrum,
	# 	path_to_parameters,
	# 	'https://api.wandb.ai/files/lewiswolf/kac_prediction%20(circular%20drum%20size)/otzl85e8/epoch_188.model',
	# )

	# # test
	# x, y = loadDataset(os.path.normpath(f'{os.path.dirname(__file__)}/data')).__getitem__(0)
	# print(f'The predicted drum size is {model(x[-1024:].view(1, -1)).item()}')
	# print(f'The true drum size is {y["drum_size"].item()}')
	# exit()
