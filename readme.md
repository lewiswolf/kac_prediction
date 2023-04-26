# kac_prediction

Deep learning toolkit for orchestrating and arranging arbitrarily shaped drums.

# Install

```bash
pip install "git+https://github.com/lewiswolf/kac_prediction.git#egg=kac_prediction"
```

### Dependencies

-	[cmake](https://formulae.brew.sh/formula/cmake)
-	[curl](https://formulae.brew.sh/formula/curl)

# Usage

The design of this library hinges on the motif of abstracting repetitive blocks of code. This is achieved by condensing the entire training routine of a single neural network into a class `Routine`. All example of usages of this class, including for other classes in this library and production use cases, can be seen in the `/scripts` directory.

<details>
<summary>Core Library</summary>

### Import

```python
from kac_prediction.pipeline import (
	# classes
	Routine,
	# types
	Datasets,
	ExportedModel,
	Model,
	Parameters,
	RunInfo,
)
```

### Classes

```python
class Routine:
	'''
	This class houses an entire training/model generation schema, and abstracts several key methods for customisability.
	Each routine should be implemented as follows.

	# initialise a default routine
	routine = Routine(exports_dir=..., wandb_config=wandb_config)

	# initialise parameters
	routine.setParameters(
		# default parameters
		SomeModel.ModelHyperParameters(*args),
		# yaml config path
		config_path=config_path,
	)

	# load, generate or install a dataset
	routine.importDataset(*args)

	# shape data
	routine.D.X = ...
	routine.D.Y = ...

	# configure model
	routine.M = SomeModel(*routine.P)

	# define how the model is to be tested
	def innerTestingLoop(i: int, loop_length: float, x: torch.Tensor, y: torch.Tensor) -> None:
		...

	# train and test a model
	routine.train(innerTestingLoop)
	'''

	device: torch.device 		# device
	D: TorchDataset				# dataset
	M: Model					# Model
	P: dict[str, Any]			# hyperparameters
	R: RunInfo					# information about the training run
	using_wandb: bool			# hidden flag for wandb

	def __init__(self, exports_dir: str = '', wandb_config: dict[str, Any] = {}) -> None:
		'''
		The init method initialises self.device, self.using_wandb and self.R.
		params:
			exports_dir		directory for the local model
			wandb_config	initialise wandb
							- None is a local call without wandb
							- passing wandb_config is a normal call
		'''

	def importDataset(
		self,
		dataset_dir: str,
		dataset_name: Datasets | Literal[''] = '',
		LocalSampler: type[AudioSampler] | None = None,
		representation_settings: RepresentationSettings = {},
		sampler_settings: SamplerSettings = {'duration': 1., 'sample_rate': 48000},
	) -> None:
		'''
		Load, download or locally generate a dataset and store it as self.D. If a dataset already exists in dataset_dir, then
		that dataset is loaded and transformed if necessary. If the project is run in evaluation mode, the official dataset
		is downloaded using the zenodo script in /bin. Else a small local dataset is generated for testing.
		'''

	def setModel(self, M: Model) -> None:
		''' Set the routine's neural network and update RunInfo. '''

	def setParameters(self, default: Parameters, config_path: str = '') -> None:
		'''
		This method initialises weights and biases if it is being used, and creates the variable self.P using either a
		default parameter value, a custom yaml file, or by being inferred from weights and biases.
		params:
			config_path		path to custom yaml file parameters
			default			default parameters
		'''

	def train(self, innerTestingLoop: Callable[[int, int, torch.Tensor, torch.Tensor], None]) -> None:
		'''
		This method runs the entire training and testing schema, beginning by splitting a dataset, running the training and
		testing loops, and exporting/saving the trained model. This method takes as its argument an innerTestingLoop(), which
		should be designed to satisfy the loop:
			for i, (x, y) in enumerate(testing_dataset):
				innerTestingLoop(i, len(testing_dataset), x.to(device), y.to(device))
		and should somewhere include the line:
			self.testing_loss[aggregate] += ...
		'''
```

### Types

```python
# string literals for installable datasets
Datasets: TypeAlias = Literal[
	'2000-convex-polygonal-drums-of-varying-size',
	'5000-circular-drums-of-varying-size',
	'5000-rectangular-drums-of-varying-dimension',
]

class ExportedModel(TypedDict, total=True):
	''' All the info needed to save and load a model. '''
	dataset: dict[str, Any]						# metadata imported from TorchDataset
	hyperparameters: dict[str, Any]				# a copy of ModelHyperParameters
	evaluation_loss: dict[str, Any] | None		# current evaluation loss, if not a test model
	model_state_dict: dict[str, Any]			# model parameters
	optimizer_state_dict: dict[str, Any]		# current optimiser state
	run_info: RunInfo							# a copy of RunInfo
	testing_loss: dict[str, Any] | None			# current testing loss if a test model
	training_loss: float						# current training loss

class Model(ABC, torch.nn.Module):
	'''
	Template for a neural network.
	'''

	criterion: torch.nn.Module
	optimiser: torch.optim.Optimizer
	testing_loss: dict[str, Any]
	training_loss: float

	@abstractmethod
	class ModelHyperParameters(Parameters):
		''' Template for custom hyper parameters. '''

	def __init__(self) -> None:

	@abstractmethod
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		''' torch.nn.Module.forward() '''

	@abstractmethod
	def innerTrainingLoop(self, i: int, loop_length: int, x: torch.Tensor, y: torch.Tensor) -> None:
		'''
		This inner training loop should be designed to satisfy the loop:
			for i, (x, y) in enumerate(training_dataset):
				Model.innerTrainingLoop(i, len(training_dataset), x.to(device), y.to(device))
		and should somewhere include the line:
			self.training_loss += ...
		'''

class ModelInfo(TypedDict, total=True):
	''' Information about the model used during a training loop. '''
	name: str									# name of the model
	version: str								# version of kac_prediction the model originates from

class Parameters(TypedDict, total=True):
	''' Default parameters for any model. '''
	batch_size: int								# batch size
	dataset_split: tuple[float, float, float]	# how is the dataset split between training, testing and evaluation
	num_of_epochs: int							# number of epochs
	testing: bool								# is the network being used for testing
	with_early_stopping: bool					# should the network stop if it has reached a minima

class RunInfo(TypedDict, total=True):
	''' Info about the current training run. '''
	epoch: int
	exports_dir: str							# absolute path to where the model should be saved locally
	id: str										# this training session's ID
	model: ModelInfo | None
```

</details>

<details>
<summary>Models</summary>

### Import

```python
from kac_prediction.architecture import CRePE
```

### Classes

```python
class CRePE(Model):
	'''
	A remake of CRePE, a deep CNN for pitch detection.
	Source: https://github.com/marl/crepe
	DOI: https://doi.org/10.48550/arXiv.1802.06182
	'''

	class ModelHyperParameters(Parameters):
		''' Template for custom hyper parameters. '''
		depth: Literal['large', 'medium', 'small', 'tiny']
		dropout: float
		learning_rate: float
		optimiser: Literal['adam', 'sgd']

	def __init__(
		self,
		depth: Literal['large', 'medium', 'small', 'tiny'],
		dropout: float,
		learning_rate: float,
		optimiser: Literal['adam', 'sgd'],
		outputs: int,
	) -> None:
		'''
		Initialise CRePE model.
		params:
			depth 			limit the size of CRePE for a trade off in accuracy.
			dropout 		hyperparameter
			learning_rate	learning rate
			optimiser 		optimiser
			outputs			number of nodes in the output layer
		'''
```

</details>

# Development / Local Training

### Dependencies

-   [pipenv](https://formulae.brew.sh/formula/pipenv#default)

### Install

```bash
git clone ...
pipenv install -d
# then for targetting gpus and linux distros
pipenv run pip install torch torchaudio --index-url ...
```

### Training Commands

Import one of the official datasets used as part of this project.

```bash
sh ./bin/install-dataset.sh 5000-circular-drums-of-varying-size
sh ./bin/install-dataset.sh 5000-rectangular-drums-of-varying-dimension
```

Then run the example training script using.

```bash
# local
pipenv run python scripts/train.py -h
	usage: train.py [-h] [--config CONFIG] [--testing] [--wandb]
	options:
	-h, --help			show this help message and exit
	--config CONFIG		path to yaml config file
	--wandb				log to wandb

# wandb
pipenv run wandb sweep scripts/config/wandb/...
pipenv run wandb agent ...
```

### Test

```bash
pipenv run test
```