'''
Example script to run inference from the command line.
'''

# if __name__ == '__main__':

# # core
# import os

# # src
# from kac_drumset import loadDataset
# from kac_prediction.architecture import CRePE
# from kac_prediction.pipeline import loadModel

# # load model
# path_to_parameters = os.path.normpath(f'{__file__}/../model/parameters.pth')
# model = loadModel(
# 	CRePE,
# 	path_to_parameters,
# 	'https://api.wandb.ai/files/lewiswolf/kac_prediction%20(circular%20drum%20size)/otzl85e8/epoch_188.model',
# )

# # test
# x, y = loadDataset(os.path.normpath(f'{os.path.dirname(__file__)}/data')).__getitem__(0)
# print(f'The predicted drum size is {model(x[-1024:].view(1, -1)).item()}')
# print(f'The true drum size is {y["drum_size"].item()}')
# exit()
