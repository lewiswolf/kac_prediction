'''
This file contains utility functions used throughout the main codebase. These functions are intended for either
debugging the package or interfacing with the file system/command line.
'''

# core
import os
import re
import shutil
import sys

__all__ = [
	# methods
	'clearDirectory',
	'printEmojis',
	# variables
	'tqdm_format',
]


# settings for the tqdm progress bar
tqdm_format = '{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}, Elapsed: {elapsed}, ETA: {remaining}, {rate_fmt}  '


def clearDirectory(absolutePath: str, excludes: list[str] = ['.gitignore']) -> None:
	'''
	Completely clears all files and folders from the input directory.
	params:
		absolutePath	absolute filepath to the directory for clearing
		excludes		files or subdirectories to exclude from clearing
	'''

	for file in os.listdir(absolutePath):
		if file in excludes:
			continue
		path = f'{absolutePath}/{file}'
		shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)


def printEmojis(s: str) -> None:
	'''
	Checks whether or not the operating system is mac or linux. If so, emojis are printed as normal, else they are
	filtered from the string.
	'''

	if sys.platform in ['linux', 'darwin']:
		print(s)
	else:
		regex = re.compile(
			'['
			u'\U00002600-\U000026FF' # miscellaneous
			u'\U00002700-\U000027BF' # dingbats
			u'\U0001F1E0-\U0001F1FF' # flags (iOS)
			u'\U0001F600-\U0001F64F' # emoticons
			u'\U0001F300-\U0001F5FF' # symbols & pictographs I
			u'\U0001F680-\U0001F6FF' # transport & map symbols
			u'\U0001F900-\U0001F9FF' # symbols & pictographs II
			u'\U0001FA70-\U0001FAFF' # symbols & pictographs III
			']+',
			flags=re.UNICODE,
		)
		print(regex.sub(r'', s).strip())
