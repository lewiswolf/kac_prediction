# core
import os
from typing import Any

# dependencies
from bokeh.embed import file_html	# convert plot to html
from bokeh.layouts import Row		# horizontal plots
from bokeh.models import Range1d	# range for plots
from bokeh.plotting import figure	# plot a figure
from bokeh.resources import CDN		# minified bokeh
import numpy as np					# math
import torch						# pytorch
import wandb						# experiment tracking

# src
from kac_drumset.samplers import PoissonModel
from kac_prediction.architecture import CRePE
from kac_prediction.dataset import RepresentationSettings
from kac_prediction.pipeline import Routine

__all__ = ['DimOfRectangularDrum']


def DimOfRectangularDrum(config_path: str = '', wandb_config: dict[str, Any] = {}) -> None:
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
			'outputs': 2,
			'testing': True,
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
		outputs=routine.P['outputs'],
	))

	# load, generate or install a dataset
	routine.importDataset(
		dataset_dir=os.path.normpath(f'{os.path.dirname(__file__)}/../../data'),
		dataset_name='' if routine.P['testing'] else '5000-rectangular-drums-of-varying-dimension',
		LocalSampler=PoissonModel,
		representation_settings=RepresentationSettings({'normalise_input': True, 'output_type': 'end2end'}),
		sampler_settings=PoissonModel.Settings({'duration': 1., 'sample_rate': 48000}),
	)

	# shape data
	routine.D.X = torch.narrow(routine.D.X, 1, 0, 1024)
	routine.D.Y = torch.tensor([[y['drum_size'], y['aspect_ratio']] for y in routine.D.Y]) # type: ignore

	# define how the model is to be tested
	def innerTestingLoop(i: int, loop_length: float, x: torch.Tensor, y: torch.Tensor) -> None:
		'''
		This method should be designed to satisfy the loop:
			for i, (x, y) in enumerate(testing_dataset):
				innerTestingLoop(i, len(testing_dataset), x.to(device), y.to(device))
		and should somewhere include the line:
			self.testing_loss[aggregate] += ...
		'''
		# calculate loss
		if i == 0:
			routine.M.testing_loss.update({
				'aspect_ratio': 0.,
				'size': 0.,
			})
		y_hat = routine.M(x)
		routine.M.testing_loss['aggregate'] += routine.M.criterion(y, y_hat).item() / loop_length
		routine.M.testing_loss['aspect_ratio'] += routine.M.criterion(y[1], y_hat[1]).item() / loop_length
		routine.M.testing_loss['size'] += routine.M.criterion(y[0], y_hat[0]).item() / loop_length
		# log to wandb
		if routine.using_wandb and i == loop_length - 1:
			# rectangle properties
			y = y.detach().cpu().numpy()[0]
			y_height = float(y[0] / (y[1] ** 0.5))
			y_width = float(y[0] * (y[1] ** 0.5))
			y_hat = np.abs(y_hat.detach().cpu().numpy()[0])
			y_hat_height = float(y_hat[0] / (y_hat[1] ** 0.5))
			y_hat_width = float(y_hat[0] * (y_hat[1] ** 0.5))
			# plots
			plot_settings: dict[str, Any] = {'height': 300, 'toolbar_location': None, 'width': 300}
			max_dim = max(2., y_width / 2., y_height / 2.)
			truth_fig = figure(
				title='Ground Truth',
				x_range=Range1d(max_dim * -1., max_dim),
				y_range=Range1d(max_dim * -1., max_dim),
				**plot_settings,
			)
			max_dim = max(2., y_hat_width / 2., y_hat_height / 2.)
			pred_fig = figure(
				title='Prediction',
				x_range=Range1d(max_dim * -1., max_dim),
				y_range=Range1d(max_dim * -1., max_dim),
				**plot_settings,
			)
			truth_fig.title.text_font = truth_fig.axis.major_label_text_font = 'CMU serif' # type: ignore
			pred_fig.title.text_font = pred_fig.axis.major_label_text_font = 'CMU serif' # type: ignore
			plot_settings = {'fill_color': '#ffffff', 'line_color': '#101010', 'x': 0., 'y': 0.}
			truth_fig.rect(width=y_width, height=y_height, **plot_settings)
			pred_fig.rect(width=y_hat_width, height=y_hat_height, **plot_settings)
			# logs
			wandb.log({
				'drum_example': wandb.Html(file_html(Row(children=[truth_fig, pred_fig]), CDN, 'Drum Example.')),
				'evaluation_loss': routine.M.testing_loss if not routine.P['testing'] else None,
				'testing_loss': routine.M.testing_loss if routine.P['testing'] else None,
				'training_loss': routine.M.training_loss,
			}, commit=True, step=routine.R['epoch'])

	# train and test a model
	routine.train(innerTestingLoop)
	wandb.finish()


if __name__ == '__main__':
	DimOfRectangularDrum(wandb_config={
		'entity': 'lewiswolf',
		'project': 'kac_prediction (rectangular drum dimension)',
	})
	exit()
