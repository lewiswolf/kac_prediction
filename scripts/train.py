'''
An example script of training a model from the CLI. This file takes advantage of the fact scripts are run upon import.
'''

if __name__ == '__main__':
	# core
	import argparse

	# initialise arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default=None, help='path to yaml config file')
	parser.add_argument('--evaluate', action='store_true', help='use evaluation dataset')
	parser.add_argument('--wandb', action='store_true', help='log to wandb')
	args = parser.parse_args()

	# switch this import statement to work with a different model.
	# from routines import DimOfRectangularDrum as training_routine
	from routines import SizeOfCircularDrum as training_routine

	# train model
	training_routine(
		config_path=args.config or '',
		testing=not args.evaluate,
		wandb_config={
			'entity': 'lewiswolf',
			'project': 'liltester',
		} if args.wandb else {},
	)
	exit()
