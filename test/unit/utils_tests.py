import sys
import os
import unittest
from unittest.mock import patch

sys.path.append('src/')  # noqa

import utils


class TestPullMicrographs(unittest.TestCase):

    def test_FileNotFound_error(self):

        # simulate a parsing error for enzyme_code = 1
        with self.assertRaises(SystemExit) as cm:
            utils.pull_micrographs(1)
        self.assertEqual(cm.exception.code, 2)


class TestPullCoordinates(unittest.TestCase):

    def test_(self):
        self.assertEqual(1, 1)


class TestDataInfo(unittest.TestCase):

    def test_(self):
        self.assertEqual(1, 1)


class TestDataExtractor(unittest.TestCase):

    def test_(self):
        self.assertEqual(1, 1)


class TestGetAllData(unittest.TestCase):

    def test_(self):
        self.assertEqual(1, 1)


if __name__ == "__main__":
    unittest.main()
