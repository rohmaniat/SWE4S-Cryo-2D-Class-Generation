'''
DELETE BEFORE MAKING PUBLIC

The purpose of this function is to look at a single enzyme file
and return two arrays:
one containing the MRC file names and the other containing the CSV file names.
'''

# python -m unittest test/unit/utils_testing.py
# TODO figure out why that command doesn't work
# run from the top level directory
# python -m unittest discover -s test/unit -p "*.py"
# use this command in terminal for now

import sys

sys.path.append('src/')  # noqa

import unittest

# import random
import utils


# This unit testing is to be executed ON OUR LOCAL MACHINES
# This file cannot be executed by GitHub Actions because it won't have
# access to the data files

class TestDataExtractor(unittest.TestCase):

    # read all the files in a folder
    # count the number of files with each extension type
    # register whether the numbers of files match up
    # register when a MRC file has no associated CSV file
    # (meaning there's no particle data)
    # I'm sure there's more

    def test_FileSearch(self):

        # there are 300 micrographs in the 10005 dataset
        self.assertEqual(utils.pull_micrographs(10005), 300)
        # there are 29 coordinate files in the 10005 dataset
        self.assertEqual(utils.pull_coordinates(10005), 29)


if __name__ == "__main__":
    unittest.main()
