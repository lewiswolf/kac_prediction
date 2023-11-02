'''
Example script to run inference from the command line.
'''

if __name__ == '__main__':
	# core
	import argparse
	import os

	# src
	from kac_prediction.architecture import CRePE
	from kac_prediction.dataset import loadDataset
	from kac_prediction.pipeline import loadModel

	# initialise arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--url', default=None, help='url to a model parameter file')
	args = parser.parse_args()

	# load a model locally or from remotely stored parameters
	# path = os.path.normpath(os.path.join(os.getcwd(), '')
	model = loadModel(CRePE, '/directory/of/parameter/files')
	model = loadModel(CRePE, '/target/a/specific/parameter/file.pt')
	model = loadModel(CRePE, '/destination/for/downloaded/parameter/file.pt', args.url)

	# test
	x, y = loadDataset(os.path.normpath(f'{os.path.dirname(__file__)}/../data')).__getitem__(0)
	print(f'The predicted drum size is {model(x[-1024:].view(1, -1)).tolist()[0][0]}')
	print(f'The true drum size is {y["drum_size"].item()}')
	exit()
