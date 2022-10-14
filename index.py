'''
A simple wrapper for `pipenv run start`. This file takes advantage of the fact scripts are run upon import.
'''

# core
import argparse


if __name__ == '__main__':
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
