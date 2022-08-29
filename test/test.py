'''
This files loads and runs all of unit tests for this project.
'''
# core
import os
import sys
import unittest

# src
sys.path.append(os.path.normpath(f'{os.path.dirname(__file__)}/../'))

# test
from src import * # noqa: F401 F403


if __name__ == '__main__':
	unittest.main()
	exit()
