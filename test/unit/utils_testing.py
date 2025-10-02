# Our data extractor is not super working so I'm going to set up some unit tests
# To hopefully figure out what is going wrong

# The purpose of this function is to look at a single enzyme file and return two arrays: 
# one containing the MRC file names and the other containing the CSV file names.

import sys

sys.path.append('src') # noqa

import unittest
import random
import numpy as np
import utils
import string

class TestDataExtractor(unittest.TestCase):
	
    # read all the files in a folder
    # count the number of files with each extension type
    # register whether the numbers of files match up
    # register when a MRC file has no associated CSV file (meaning there's no particle data)
    # I'm sure there's more
    
    def test_FileSearch(self):

        self.assertEqual (utils.data_extractor([1, 2, 3, 4, 5]), 3)
        self.assertEqual (my_utils.mean([10, 20, 30, 40, 50]), 30)