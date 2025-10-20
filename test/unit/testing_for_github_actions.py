import sys
import unittest
from unittest import mock
import mrcfile

sys.path.append('src/')  # noqa

# import random
import utils


# This unit testing WILL be executed by GitHub actions
class TestDataExtractor(unittest.TestCase):

    def test_FileNotFound_error(self):

        self.assertEqual(1, 1)  # dummy test I'll delete soon

        # simulate a parsing error for enzyme_code = 1
        with self.assertRaises(SystemExit) as cm:
            utils.pull_micrographs(1)
        self.assertEqual(cm.exception.code, 2)


class TestParticlePicker(unittest.TestCase):

    def test_(self):

        # no clue how to test this if the data is outside out repo
        self.assertEqual(1, 1)


if __name__ == "__main__":
    unittest.main()
