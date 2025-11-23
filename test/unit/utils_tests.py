import sys
import os
import unittest
from unittest.mock import patch

sys.path.append('src/')  # noqa

import utils


class TestPullMicrographs(unittest.TestCase):

    @patch('os.listdir')
    @patch('os.path.isfile')

    def test_FileNotFound(self, mock_isfile, mock_listdir):
        # make sure that None is raised when no MRC files are found

        result = utils.pull_micrographs(0)
        self.assertEqual(result, None)

    @patch('os.listdir')
    @patch('os.path.isfile')

    def test_pull_micrograph_files(self, mock_isfile, mock_listdir):
        # using a mock directory, check that pull_micrographs file
        # counting is working

        mock_listdir.return_value = [
            'file1.mrc', 'file2.mrc', 'file3.mrc', 'file4.mrc']
        mock_isfile.return_value = True

        result = utils.pull_micrographs('enzymeA')
        self.assertEqual(result, 4)

    @patch('os.listdir')
    @patch('os.path.isfile')

    def test_pull_micrograph_nofiles(self, mock_isfile, mock_listdir):
        # when there are no files, there should be a ValueError

        mock_listdir.return_value = []
        mock_isfile.return_value = False

        result = utils.pull_micrographs('enzymeB')
        self.assertEqual(result, None)


class TestPullCoordinates(unittest.TestCase):

    @patch('os.listdir')
    @patch('os.path.isfile')

    def test_withfiles(self, mock_isfile, mock_listdir):
        # make sure that pull_coordinates is reading the csv files

        mock_listdir.return_value = ['coords1.csv', 'coords2.csv']
        mock_isfile.return_value = True

        result = utils.pull_coordinates('enzymeC')
        self.assertEqual(result, 2)

    @patch('os.listdir')
    @patch('os.path.isfile')

    def test_empty_directory(self, mock_isfile, mock_listdir):
        # test pull_coordinates on an empty directory

        mock_listdir.return_value = []
        mock_isfile.return_value = False

        result = utils.pull_coordinates('enzymeD')
        self.assertEqual(result, 0)

    @patch('os.listdir')
    @patch('os.path.isfile')

    def test_FileNotFound(self, mock_isfile, mock_listdir):
        # Setup the mock to raise a FileNotFoundError

        mock_listdir.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            utils.pull_coordinates('enzymeE')


class TestDataInfo(unittest.TestCase):

    @patch("os.listdir")

    def test_data_info(self, mock_listdir):
        # test data_info in parsing CSV and MRC files

        # make a mock list of MRC and CSV files
        mock_listdir.side_effect = [
            ["a.mrc", "b.mrc", "c.mrc"],       # micrographs
            ["a.csv", "b.csv", "c.csv"]]       # coordinates
        
        imgs, csvs = utils.data_info("EnzymeF")

        self.assertEqual(imgs, ["a.mrc", "b.mrc", "c.mrc"])
        self.assertEqual(csvs, ["a.csv", "b.csv", "c.csv"])

    @patch("os.listdir")

    def test_empty_folders(self, mock_listdir):
        # test the function on empty file lists
        # should show up empty (which is handled later in the code)
        # this function should just return empty lists

        mock_listdir.side_effect = [[], []]

        imgs, csvs = utils.data_info("EnzymeG")
        self.assertEqual(imgs, [])
        self.assertEqual(csvs, [])


class TestDataExtractor(unittest.TestCase):

    @patch('os.listdir')

    def test_missing_image_dir(self, mock_exists):
        # Make image_dir return False, csv_dir return True
        mock_exists.side_effect = [False, True]

        result = utils.data_extractor("EnzymeH")
        self.assertIsNone(result)

    @patch('os.listdir')

    def test_missing_csv_dir(self, mock_exists):
        # Make image_dir return True, csv_dir return False
        mock_exists.side_effect = [True, False]

        result = utils.data_extractor("EnzymeI")
        self.assertIsNone(result)

    @patch('os.listdir')
    @patch("utils.os.path.exists")

    def test_extractor(self, mock_exists, mock_listdir):
        # All MRC files should have a CSV file of the same name
        # Organize and return this data

        mock_exists.side_effect = [True, True]
        mock_listdir.side_effect = [
            ["a.mrc", "b.mrc"],
            ["a.csv", "b.csv"]]
        
        result = utils.data_extractor("EnzymeJ")
        self.assertEqual(result, (["a.mrc", "b.mrc"], ["a.csv", "b.csv"]))

    @patch('os.listdir')
    @patch("utils.os.path.exists")

    def test_extractor_matching(self, mock_exists, mock_listdir):
        # if there are MRC files with no CSV, remove them from analysis

        mock_exists.side_effect = [True, True]
        mock_listdir.side_effect = [
            ["a.mrc", "b.mrc", "c.mrc"],
            ["a.csv", "b.csv"]]
        
        result = utils.data_extractor("EnzymeK")
        self.assertEqual(result, (["a.mrc", "b.mrc"], ["a.csv", "b.csv"]))


class TestGetAllData(unittest.TestCase):

    @patch("utils.data_extractor")
    @patch("utils.os.getcwd")
    @patch("utils.os.listdir")

    def test_find_all_data_normal(
        self, mock_listdir, mock_getcwd, mock_extractor):
    # making a mock filepath for EnzymeL and EnzymeM
    # should return matching lists for both

        mock_getcwd.return_value = "/home/user/project"
        mock_listdir.return_value = ["enzymeL", "enzymeM"]

        mock_extractor.side_effect = [
            (["a1.mrc", "a2.mrc"], ["a1.csv", "a2.csv"]),
            (["b1.mrc"], ["b1.csv"])]

        mrc, csv = utils.find_all_data()

        expected_mrc = [
            "/home/user/project/../Data/enzymeL/micrographs/a1.mrc",
            "/home/user/project/../Data/enzymeL/micrographs/a2.mrc",
            "/home/user/project/../Data/enzymeM/micrographs/b1.mrc",
        ]

        expected_csv = [
            "/home/user/project/../Data/enzymeL/ground_truth/particle_coordinates/a1.csv",
            "/home/user/project/../Data/enzymeL/ground_truth/particle_coordinates/a2.csv",
            "/home/user/project/../Data/enzymeM/ground_truth/particle_coordinates/b1.csv",
        ]

        self.assertEqual(mrc, expected_mrc)
        self.assertEqual(csv, expected_csv)

    @patch("utils.data_extractor")
    @patch("utils.os.getcwd")
    @patch("utils.os.listdir")

    def test_find_data_skip_nones(self, mock_listdir, mock_getcwd, mock_extractor):
        # if data_extractor returns None, skip that enzyme
        # skip EnzymeN, return EnzymeO

        mock_getcwd.return_value = "/cwd"
        mock_listdir.return_value = ["enzymeN", "enzymeO"]

        mock_extractor.side_effect = [
            None,                            
            (["b1.mrc"], ["b1.csv"])]

        mrc, csv = utils.find_all_data()

        self.assertEqual(
            mrc,
            ["/cwd/../Data/enzymeO/micrographs/b1.mrc"])
        self.assertEqual(
            csv,
            ["/cwd/../Data/enzymeO/ground_truth/particle_coordinates/b1.csv"])

    @patch("utils.data_extractor")
    @patch("utils.os.getcwd")
    @patch("utils.os.listdir")

    def test_extractor_exception(self, mock_listdir, mock_getcwd, mock_extractor):
        # If data_extractor raises an exception, skip that enzyme
        # skip EnzymeP, continue with EnzymeQ

        mock_getcwd.return_value = "/cwd"
        mock_listdir.return_value = ["enzymeP", "enzymeQ"]

        mock_extractor.side_effect = [
            Exception("any exception"),
            (["b1.mrc"], ["b1.csv"])]

        mrc, csv = utils.find_all_data()

        self.assertEqual(
            mrc,
            ["/cwd/../Data/enzymeQ/micrographs/b1.mrc"])
        self.assertEqual(
            csv,
            ["/cwd/../Data/enzymeQ/ground_truth/particle_coordinates/b1.csv"])
        
    @patch("utils.data_extractor")
    @patch("utils.os.getcwd")
    @patch("utils.os.listdir")

    def test_find_all_data_empty(self, mock_listdir, mock_getcwd, mock_extractor):
        # reutrn empty lists if there are no files

        mock_getcwd.return_value = "/cwd"
        mock_listdir.return_value = []
        mock_extractor.return_value = None

        mrc, csv = utils.find_all_data()

        self.assertEqual(mrc, [])
        self.assertEqual(csv, [])


if __name__ == "__main__":
    unittest.main()
