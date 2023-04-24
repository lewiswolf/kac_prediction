# core
import os
from typing import Any

# dependencies
from bokeh.embed import file_html	# convert plot to html
from bokeh.layouts import Row		# horizontal plots
from bokeh.plotting import figure	# plot a figure
from bokeh.resources import CDN		# minified bokeh
import torch						# pytorch
import wandb						# experiment tracking

# src
from kac_drumset import BesselModel, RepresentationSettings
from kac_prediction.architecture import CRePE
from kac_prediction.pipeline import Routine

__all__ = ['SizeOfCircularDrum']


def SizeOfCircularDrum(config_path: str = '', testing: bool = True, wandb_config: dict[str, Any] = {}) -> None:
	'''
	Perform the entire training routine
	'''

	# initialise a default routine
	routine = Routine(
		exports_dir=os.path.normpath(f'{os.path.dirname(__file__)}/../../model'),
		wandb_config=wandb_config,
	)

	# initialise parameters
	routine.setParameters(
		# default parameters
		CRePE.ModelHyperParameters({
			'batch_size': 5,
			'dataset_split': (0.7, 0.15, 0.15),
			'depth': 'tiny',
			'dropout': 0.25,
			'learning_rate': 1e-3,
			'num_of_epochs': 50,
			'optimiser': 'sgd',
			'testing': testing,
			'with_early_stopping': True,
		}),
		# yaml config path
		config_path=config_path,
	)

	# configure model
	routine.setModel(CRePE(
		depth=routine.P['depth'],
		dropout=routine.P['dropout'],
		learning_rate=routine.P['learning_rate'],
		optimiser=routine.P['optimiser'],
		outputs=1,
	))

	# load, generate or install a dataset
	routine.importDataset(
		dataset_dir=os.path.normpath(f'{os.path.dirname(__file__)}/../../data'),
		dataset_name='' if routine.P['testing'] else '5000-circular-drums-of-varying-size',
		LocalSampler=BesselModel,
		representation_settings=RepresentationSettings({'normalise_input': True, 'output_type': 'end2end'}),
		sampler_settings=BesselModel.Settings({'duration': 1., 'sample_rate': 48000}),
	)

	# shape data
	routine.D.X = torch.narrow(routine.D.X, 1, 0, 1024)
	routine.D.Y = torch.tensor([[y['drum_size']] for y in routine.D.Y]) # type: ignore

	# define how the model is to be tested
	def innerTestingLoop(i: int, loop_length: float, x: torch.Tensor, y: torch.Tensor) -> None:
		'''
		This method should be designed to satisfy the loop:
			for i, (x, y) in enumerate(testing_dataset):
				innerTestingLoop(i, len(testing_dataset), x.to(device), y.to(device))
		and should somewhere include the line:
			self.testing_loss += ...
		'''
		# calculate loss
		y_hat = routine.M(x)
		routine.M.testing_loss['aggregate'] += routine.M.criterion(y, y_hat).item() / loop_length
		# log to wandb
		if routine.using_wandb and i == loop_length - 1:
			# plots
			plot_settings: dict[str, Any] = {'height': 300, 'width': 300}
			truth_fig = figure(title='Ground Truth', **plot_settings)
			pred_fig = figure(title='Prediction', **plot_settings)
			plot_settings = {'fill_color': '#1B9E31', 'line_color': '#126B21', 'x': 0., 'y': 0.}
			truth_fig.circle(radius=y.detach().cpu().numpy()[0] / 2, **plot_settings)
			pred_fig.circle(radius=y_hat.detach().cpu().numpy()[0] / 2, **plot_settings)
			# logs
			wandb.log({
				'drum_example': wandb.Html(file_html(Row(children=[truth_fig, pred_fig]), CDN, 'Drum Example.')),
				'epoch': routine.R['epoch'],
				'evaluation_loss': routine.M.testing_loss['aggregate'] if not routine.P['testing'] else None,
				'testing_loss': routine.M.testing_loss['aggregate'] if routine.P['testing'] else None,
				'training_loss': routine.M.training_loss,
			}, commit=True)

	# train and test a model
	routine.train(innerTestingLoop)
	wandb.finish()


if __name__ == '__main__':
	SizeOfCircularDrum(wandb_config={
		'entity': 'lewiswolf',
		'project': 'kac_prediction (circular drum size)',
	})
	exit()
