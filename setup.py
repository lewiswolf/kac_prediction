'''
Build the kac_prediction module.
'''

# core
import codecs
import os
from setuptools import find_packages, setup


this = os.path.abspath(os.path.dirname(__file__))
name = 'kac_prediction'
version = '1.1.0'
short_description = 'Neural networks designed for working with arbitrarily shaped drums.'

# import long description from readme.md
with codecs.open(os.path.join(this, 'readme.md'), encoding='utf-8') as readme:
	long_description = '\n' + readme.read()

# import packages from Pipfile
with codecs.open(os.path.join(this, 'Pipfile'), encoding='utf-8') as raw_pipfile:
	packages = []
	# read the Pipfile
	pipfile = raw_pipfile.readlines(1)
	raw_pipfile.close()
	# loop over the file
	is_pkg = False
	for line in pipfile:
		line = line.replace('\n', '')
		if not line:
			continue
		# find [packages]
		if line[0] == '[':
			is_pkg = line == '[packages]'
			continue
		# append package names with required version / git config
		if is_pkg:
			pkg_name, _, *spec = line.split()
			if spec[0] == '"*"' or spec[0] == '{file' or spec[0] == '{path':
				packages.append(pkg_name)
			elif spec[0] == '{git':
				packages.append(f'{pkg_name} @ git+{spec[2][1:-2]}')
			else:
				packages.append(f'{pkg_name}{spec[0][1:-1]}')

setup(
	author='Lewis Wolstanholme',
	author_email='lewiswolstanholme@gmail.com',
	classifiers=[
		'Operating System :: MacOS :: MacOS X',
		'Operating System :: Microsoft :: Windows',
		'Operating System :: Unix',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3 :: Only',
		'Programming Language :: Python :: 3.11',
		'Programming Language :: Python :: 3.12',
		'Programming Language :: Python :: 3.13',
		'Typing :: Typed',
	],
	description=short_description,
	install_requires=packages,
	keywords=['kac_prediction'],
	long_description=long_description,
	long_description_content_type='text/markdown',
	name=name,
	package_dir={'kac_prediction': 'kac_prediction'},
	package_data={'kac_prediction': ['py.typed']},
	packages=find_packages(),
	version=version,
)
