'''
The main method for training a neural network.
'''

# dependencies
import torch		# pytorch

# src
from kac_drumset.utils import printEmojis

__all__ = [
	'trainModel',
]


def trainModel(
	trainingDataset: torch.utils.data.DataLoader,
	testingDataset: torch.utils.data.DataLoader,
) -> None:
	# ) -> tuple[nn.Module, float]:
	'''
	Training loop for a neural network
	'''

	# configure device
	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		printEmojis('😓 WARNING 😓 Nvidia GPU support is not available for training the network.')
		device = torch.device('cpu')

	printEmojis('Training neural network... 🧠')
	device
