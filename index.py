'''
A simple wrapper for `pipenv run start`. This file takes advantage of the fact scripts are run upon import.
'''


if __name__ == '__main__':
	# switch the import statement to a different routine to work with a different model.
	import kac_prediction.routine.size_of_circular_drum # noqa: F401
	exit()
