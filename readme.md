# kac_prediction

Deep learning toolkit for orchestrating and arranging arbitrarily shaped drums.

# Install

```bash
pip install "git+https://github.com/lewiswolf/kac_prediction.git"
```

### Dependencies

-	[cmake](https://formulae.brew.sh/formula/cmake)
-	[curl](https://formulae.brew.sh/formula/curl)

# Usage

The design of this library hinges on the motif of abstracting repetitive blocks of code. This is achieved by condensing the entire training routine of a single neural network into a class `Routine`. All example of usages of this class, including for other classes in this library and production use cases, can be seen in the `/scripts` directory. Similarly, this package comes equipped with a customisable class `AudioSampler`, which can be used to create a dataset of synthesised audio samples.

<details>
<summary>Core Library</summary>

### Import

```python
from kac_prediction.pipeline import (
	# Classes
	Routine,
	# Types
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
	routine = Routine(
		# directory to store saved models
		exports_dir=...,
		# configuration for wandb
		wandb_config=...,
	)

	# initialise parameters
	routine.setParameters(
		# default parameters
		SomeModel.ModelHyperParameters(*args),
		# yaml config path
		config_path=...,
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
	''' Info about the current routine. '''
	epoch: int									# epoch during training or of loaded parameters
	exports_dir: str							# absolute path to where the model should be saved locally
	id: str										# this training session's ID
	model: ModelInfo | None						# info about the neural model being used
```

</details>

<details>
<summary>Dataset</summary>

### Import

```python
from kac_prediction.dataset import (
	# Methods
	generateDataset,
	loadDataset,
	regenerateDataPoints,
	transformDataset,
	# Classes
	AudioSampler,
	InputRepresentation,
	# Types
	RepresentationSettings,
	SamplerInfo,
	SamplerSettings,
	TorchDataset,
)
```

### Methods

```python
def generateDataset(
	Sampler: Type[AudioSampler],
	sampler_settings: SamplerSettings,
	dataset_dir: str,
	dataset_size: int = 10,
	representation_settings: RepresentationSettings | None = None,
) -> TorchDataset:
	'''
	Generates a dataset of audio samples. The generated dataset, including the individual .wav files and the metadata.json,
	are saved in the directory specified by the absolute filepath dataset_dir.
	'''

def loadDataset(dataset_dir: str) -> TorchDataset:
	'''
	loadDataset imports a kac_drumset dataset from the directory specified by the absolute path dataset_dir.
	'''

def regenerateDataPoints(dataset: TorchDataset, Sampler: type[AudioSampler], entries: list[int]) -> TorchDataset:
	'''
	This method regenerates specific indices of a dataset.
	'''

def transformDataset(dataset: TorchDataset, representation_settings: RepresentationSettings) -> TorchDataset:
	'''
	transformDataset is used to transform the input representation of a loaded dataset. This method rewrites the
	metadata.json for the dataset, such that the dataset will be loaded with the new settings upon future use.
	'''
```

### Classes

```python
class AudioSampler(ABC):
	''' Abstract parent class for an audio sampler. '''

	duration: float						# duration of the audio file (seconds)
	length: int							# length of the audio file (samples)
	sample_rate: int					# sample rate
	waveform: npt.NDArray[np.float64]	# the audio sample itself

	def __init__(self, duration: float, sample_rate: int, any_other_custom_kwargs: Any) -> None:
		'''
		When defining a custom audio sampler, you must call the below method so as to save your custom kwargs as part of 
		the metadata.json. In special cases, where the kwarg type is not a basic type (float, string, etc.), you may wish
		to amend the return value of classLocalsToKwargs.
		'''
		from kac_prediction.dataset import classLocalsToKwargs
		super().__init__(**classLocalsToKwargs(locals()))

	def export(self, absolutePath: str, bit_depth: Literal[16, 24, 32] = 24) -> None:
		''' Write the generated waveform to a .wav file. '''

	@abstractmethod
	def generateWaveform(self) -> None:
		''' This method should be used to generate and set self.waveform. '''

	@abstractmethod
	def getLabels(self) -> dict[str, list[float | int]]:
		''' This method should return the y labels for the generated audio. '''

	@abstractmethod
	def updateProperties(self, i: int | None]) -> None:
		''' This method should be used to update the properties of the sampler when inside a generator loop. '''

	@abstractmethod
	class Settings(SamplerSettings, total=False):
		'''
		This is an abstract TypedDict used to mirror the type declaration for the customised __init__() method. This allows
		for type safety when using a custom AudioSampler with an arbitrary __init__() method.
		'''

class InputRepresentation():
	'''
	This class is used to convert a raw waveform into a user defined input representation, which includes end2end, the
	fourier transform, and a mel spectrogram.
	'''

	settings: RepresentationSettings

	def __init__(self, sample_rate: int, settings: RepresentationSettings | None = None) -> None:
		'''
		InputRepresentation works by creating a variably defined method self.transform. This method uses the input settings to
		generate the correct input representation of the data.
		'''

	def transform(self, waveform: npt.NDArray[np.float64]) -> torch.Tensor:
		''' Produce the output representation. '''

	@staticmethod
	def normalise(waveform: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
		''' Normalise an audio waveform, such that x ∈ [-1.0, 1.0] '''

	@staticmethod
	def transformShape(data_length: int, settings: RepresentationSettings) -> tuple[int, ...]:
		''' This method uses the length of the incoming audio data to calculate the size of the transform's output. '''
```

### Types

```python
class RepresentationSettings(TypedDict, total=False):
	'''
	These settings are used to specify the data representation of audio, providing the option for end to end data, as well
	as Fourier and Mel transformations. An FFT is calculated using n_bins for the number of frequency bins, as well as
	window_length and hop_length for the size of the bins. The Mel representation uses the same settings as the FFT, with
	the addition of n_mels, the number of mel frequency bins, and f_min, the minimum frequency of the transform.
	'''

	f_min: float			# minimum frequency of the transform in hertz (mel only)
	hop_length: int			# hop length in samples
	n_bins: int				# number of frequency bins for the spectral density function
	n_mels: int				# number of mel frequency bins (mel only)
	normalise_input: bool	# should the input be normalised
	output_type: Literal[	# representation type
		'end2end',
		'fft',
		'mel',
	]
	window_length: int		# window length in samples

class SamplerInfo(TypedDict, total=True):
	'''
	Information about the sampler used to generate a specific dataset.
	'''
	name: str		# name of the sampler
	version: str	# version of kac_drumset when the sampler was generated

class SamplerSettings(TypedDict, total=True):
	'''
	These are the minimum requirements for the AudioSampler __init__() method. This type is used to maintain type safety
	when using a custom AudioSampler.
	'''
	duration: float		# duration of the audio file (seconds)
	sample_rate: int	# sample rate

class TorchDataset(torch.utils.data.Dataset):
	''' PyTorch wrapper for a dataset. '''

	dataset_dir: str									# dataset directory
	representation_settings: RepresentationSettings		# settings for InputRepresentation
	sampler: SamplerInfo								# the name of the sampler used to generate the dataset
	sampler_settings: dict[str, Any]					# settings for the sampler
	X: torch.Tensor										# data
	Y: list[dict[str, torch.Tensor]]					# labels

	def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
		''' Return the data and its labels at index i. '''

	def __len__(self) -> int:
		''' Return the dataset size. '''
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

<details><summary>Samplers</summary>

### Import

```python
from kac_prediction.samplers import (
	# Types
	Datasets
	# Samplers
	TestSweep,
	TestTone,
)
```

### Types

```python
# string literals for installable datasets
Datasets: TypeAlias = Literal[
	'2000-convex-polygonal-drums-of-varying-size',
	'5000-circular-drums-of-varying-size',
	'5000-rectangular-drums-of-varying-dimension',
]
```

### Samplers

```python
class TestSweep(AudioSampler):
	'''
	This class produces a sine wave sweep across the audio spectrum, from 20hz to f_s / 2.
	'''
		
class TestTone(AudioSampler):
	'''
	This class produces an arbitrary test tone, using either a sawtooth, sine, square or triangle waveform. If it's initial frequency is not set, it will automatically create random frequencies.
	'''

	class Settings(SamplerSettings, total=False):
		f_0: float										# fixed fundamental frequency (hz)
		waveshape: Literal['saw', 'sin', 'sqr', 'tri']	# shape of the waveform
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
pipenv run python scripts/train.py
	usage: train.py [-h] [--config CONFIG] [--wandb]
	options:
	-h, --help			show this help message and exit
	--config CONFIG		path to yaml config file
	--wandb				log to wandb

# wandb
pipenv run wandb sweep scripts/config/wandb/...
pipenv run wandb agent ...
```

### Deploy a Trained Model

```bash
pipenv run python scripts/deploy.py
```

### Test Codebase

```bash
pipenv run test
```

# Related Works

```bibtex
@inproceedings{wolstanholmeOrchestratingPhysicallyModelled2023,
  title = {Towards Orchestrating Physically Modelled {{2D}} Percussion Instruments},
  booktitle = {10th {{Convention}} of the {{European Acoustics Association}} ({{Forum Acusticum}})},
  author = {Wolstanholme, Lewis and McPherson, Andrew},
  year = {2023},
  month = sep,
  publisher = {{Turin, Italy}},
  copyright = {All rights reserved}
}

@misc{wolstanholmeKac_drumsetDatasetGenerator2022,
  title = {{{kac\_drumset}}: {{A}} Dataset Generator for Arbitrarily Shaped Drums},
  author = {Wolstanholme, Lewis},
  year = {2022},
  month = nov,
  publisher = {{Zenodo}},
  doi = {10.5281/zenodo.7057219},
  copyright = {All rights reserved}
}
```